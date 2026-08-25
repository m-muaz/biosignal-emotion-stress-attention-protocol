"""Participant-level train/val/test splitting.

Physiological windows must never be split independently of the participant
they came from (plan §40-41) -- this module only ever splits a list of
participant IDs, never rows/windows, and provides the disjointness check
every dataset manifest should carry.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class ParticipantSplit:
    train: list[str]
    val: list[str]
    test: list[str]

    def to_dict(self) -> dict:
        return {"train": self.train, "val": self.val, "test": self.test}

    def assert_disjoint(self) -> None:
        train, val, test = set(self.train), set(self.val), set(self.test)
        assert train.isdisjoint(val), f"train/val overlap: {train & val}"
        assert train.isdisjoint(test), f"train/test overlap: {train & test}"
        assert val.isdisjoint(test), f"val/test overlap: {val & test}"


def split_by_participant(
    participant_ids: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> ParticipantSplit:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "ratios must sum to 1.0"
    ids = sorted(set(participant_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    # remainder goes to test, so every participant is placed even under rounding
    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]

    split = ParticipantSplit(train=train, val=val, test=test)
    split.assert_disjoint()
    return split
