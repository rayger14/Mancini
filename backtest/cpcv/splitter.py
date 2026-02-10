"""Combinatorial Purged Cross-Validation splitter for day-grouped data."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from math import comb
from typing import Iterator


@dataclass(frozen=True)
class CPCVConfig:
    """Configuration for CPCV splitting."""

    n_groups: int = 10
    n_test_groups: int = 2
    purge_days: int = 2
    embargo_days: int = 1


@dataclass
class CPCVSplit:
    """A single train/test split from CPCV."""

    split_id: int
    test_group_indices: tuple[int, ...]
    train_dates: list[date]
    test_dates: list[date]
    purged_dates: list[date]
    embargoed_dates: list[date]


class CPCVSplitter:
    """Combinatorial Purged Cross-Validation at the day-group level.

    Divides N trading days into ``n_groups`` contiguous groups.
    Generates C(n_groups, n_test_groups) splits, each using
    ``n_test_groups`` groups for testing and the rest for training.

    Purging removes ``purge_days`` trading days from the training set
    at each boundary between a train group and an adjacent test group
    (bidirectional, because prior_day_df creates a 1-day leak path).

    Embargo removes ``embargo_days`` training days after each test group.
    """

    def __init__(
        self,
        trading_days: list[date],
        config: CPCVConfig = CPCVConfig(),
    ) -> None:
        self.config = config
        self.trading_days = sorted(trading_days)
        self.n_days = len(self.trading_days)
        self.groups: list[list[date]] = self._make_groups()
        self.n_splits = comb(config.n_groups, config.n_test_groups)

    @property
    def num_paths(self) -> int:
        return self.n_splits

    def _make_groups(self) -> list[list[date]]:
        n = self.config.n_groups
        base_size = self.n_days // n
        remainder = self.n_days % n
        groups: list[list[date]] = []
        idx = 0
        for g in range(n):
            size = base_size + (1 if g < remainder else 0)
            groups.append(self.trading_days[idx : idx + size])
            idx += size
        return groups

    def splits(self) -> Iterator[CPCVSplit]:
        """Yield all CPCV splits with purging and embargo applied."""
        all_days_set = set(self.trading_days)

        for split_id, test_combo in enumerate(
            combinations(range(self.config.n_groups), self.config.n_test_groups)
        ):
            test_dates_set: set[date] = set()
            for g_idx in test_combo:
                test_dates_set.update(self.groups[g_idx])

            purged: set[date] = set()
            embargoed: set[date] = set()

            for g_idx in test_combo:
                test_group = self.groups[g_idx]
                test_start = test_group[0]
                test_end = test_group[-1]

                # Purge: remove train days before and after test group
                purged.update(self._n_days_before(test_start, self.config.purge_days))
                purged.update(self._n_days_after(test_end, self.config.purge_days))

                # Embargo: remove train days after test group
                embargoed.update(self._n_days_after(test_end, self.config.embargo_days))

            remove_from_train = (purged | embargoed) - test_dates_set
            train_dates_set = all_days_set - test_dates_set - remove_from_train

            yield CPCVSplit(
                split_id=split_id,
                test_group_indices=test_combo,
                train_dates=sorted(train_dates_set),
                test_dates=sorted(test_dates_set),
                purged_dates=sorted(purged - test_dates_set),
                embargoed_dates=sorted(embargoed - test_dates_set),
            )

    def _n_days_before(self, target: date, n: int) -> list[date]:
        idx = self._day_index(target)
        if idx is None:
            return []
        start = max(0, idx - n)
        return self.trading_days[start:idx]

    def _n_days_after(self, target: date, n: int) -> list[date]:
        idx = self._day_index(target)
        if idx is None:
            return []
        end = min(self.n_days, idx + 1 + n)
        return self.trading_days[idx + 1 : end]

    def _day_index(self, d: date) -> int | None:
        i = bisect_left(self.trading_days, d)
        if i < self.n_days and self.trading_days[i] == d:
            return i
        return None

    def summary(self) -> str:
        return (
            f"CPCV: {self.n_days} days -> {self.config.n_groups} groups "
            f"(~{self.n_days // self.config.n_groups} days/group)\n"
            f"Test groups per split: {self.config.n_test_groups}\n"
            f"Total splits: {self.n_splits}\n"
            f"Purge: {self.config.purge_days} days, "
            f"Embargo: {self.config.embargo_days} days"
        )
