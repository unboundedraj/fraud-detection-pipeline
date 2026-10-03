"""Feature definitions shared by batch (SQL) and streaming (Python) paths."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field

PCA_COLUMNS = [f"v{i}" for i in range(1, 29)]
ENGINEERED_COLUMNS = [
    "amount_log",
    "hour_of_day",
    "txn_count_prev_window",
    "amount_avg_prev_window",
    "amount_zscore_prev_window",
]
FEATURE_COLUMNS = PCA_COLUMNS + ENGINEERED_COLUMNS
LABEL_COLUMN = "is_fraud"


@dataclass
class OnlineFeatureState:
    """Incremental equivalent of the window features in sql/gold_features.sql.

    Holds the (event_ts, amount) history needed for the longest window and answers
    "what did the last N seconds look like, *excluding* the current second?" in
    O(log n) via bisect on timestamps plus prefix sums. Events must arrive in
    non-decreasing event time (the stream producer replays them that way; late events
    are rejected by the consumer and routed to a dead-letter sink).
    """

    velocity_window_s: int = 60
    amount_window_s: int = 3600
    _ts: list[float] = field(default_factory=list)
    _cum_amount: list[float] = field(default_factory=lambda: [0.0])
    _cum_amount_sq: list[float] = field(default_factory=lambda: [0.0])
    _offset: int = 0  # number of evicted events (keeps prefix sums valid after compaction)

    @property
    def watermark(self) -> float | None:
        return self._ts[-1] if self._ts else None

    def _range_stats(self, start_ts: float, end_ts_exclusive: float) -> tuple[int, float, float]:
        lo = bisect_left(self._ts, start_ts)
        hi = bisect_left(self._ts, end_ts_exclusive)
        n = hi - lo
        s = self._cum_amount[hi] - self._cum_amount[lo]
        sq = self._cum_amount_sq[hi] - self._cum_amount_sq[lo]
        return n, s, sq

    def compute(self, event_ts: float, amount: float) -> dict[str, float]:
        """Features for an event at epoch-seconds `event_ts` (state is not modified)."""
        # Window bounds mirror SQL: [ts - window, ts - 1s]  ==  [ts - window, ts) for
        # whole-second timestamps.
        n_vel, _, _ = self._range_stats(event_ts - self.velocity_window_s, event_ts)
        n_amt, s, sq = self._range_stats(event_ts - self.amount_window_s, event_ts)

        mean = s / n_amt if n_amt else 0.0
        zscore = 0.0
        if n_amt >= 2:
            var = (sq - n_amt * mean * mean) / (n_amt - 1)
            std = math.sqrt(var) if var > 0 else 0.0
            zscore = (amount - mean) / std if std > 0 else 0.0

        hour = int((event_ts // 3600) % 24)
        return {
            "amount_log": math.log1p(amount),
            "hour_of_day": hour,
            "txn_count_prev_window": n_vel,
            "amount_avg_prev_window": mean,
            "amount_zscore_prev_window": zscore,
        }

    def update(self, event_ts: float, amount: float) -> None:
        if self._ts and event_ts < self._ts[-1]:
            raise ValueError(f"out-of-order event {event_ts} < watermark {self._ts[-1]}")
        self._ts.append(event_ts)
        self._cum_amount.append(self._cum_amount[-1] + amount)
        self._cum_amount_sq.append(self._cum_amount_sq[-1] + amount * amount)
        if len(self._ts) > 50_000:
            self._compact()

    def _compact(self) -> None:
        """Drop events older than the longest window to bound memory."""
        horizon = self._ts[-1] - max(self.velocity_window_s, self.amount_window_s) - 1
        cut = bisect_left(self._ts, horizon)
        if cut == 0:
            return
        base_a, base_sq = self._cum_amount[cut], self._cum_amount_sq[cut]
        self._ts = self._ts[cut:]
        self._cum_amount = [c - base_a for c in self._cum_amount[cut:]]
        self._cum_amount_sq = [c - base_sq for c in self._cum_amount_sq[cut:]]
        self._offset += cut
