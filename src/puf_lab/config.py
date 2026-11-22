"""Configuration for the simulation and the fixed-budget experiment."""

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int
    challenge_bits: int
    chains: tuple[int, ...]
    noise_levels: tuple[float, ...]
    devices: int
    reference_challenges: int
    uniqueness_challenges: int
    budgets: tuple[int, ...]
    enrollment_reads: tuple[int, ...]
    trials: int
    bootstrap_resamples: int

    @classmethod
    def load(cls, path: Path):
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        for field in ("chains", "noise_levels", "budgets", "enrollment_reads"):
            data[field] = tuple(data[field])
        config = cls(**data)
        config.validate()
        return config

    def validate(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        positive_fields = (
            self.challenge_bits, self.devices, self.reference_challenges,
            self.uniqueness_challenges, self.trials, self.bootstrap_resamples,
        )
        if any(type(value) is not int or value < 2 for value in positive_fields):
            raise ValueError("sample sizes, challenge length and resamples must be integers >= 2")
        if self.devices < 4 or self.devices % 2:
            raise ValueError("devices must be even and >= 4")
        for name, values in (
            ("chains", self.chains), ("budgets", self.budgets),
            ("enrollment_reads", self.enrollment_reads),
        ):
            if not values or any(type(value) is not int or value < 1 for value in values):
                raise ValueError(f"{name} must contain positive integers")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be strictly increasing")
        if self.enrollment_reads[0] != 1 or any(r % 2 == 0 for r in self.enrollment_reads):
            raise ValueError("enrollment_reads must start with 1 and contain only odd values")
        if any(budget % (r + 1) for budget in self.budgets for r in self.enrollment_reads):
            raise ValueError("each budget must be divisible by every enrollment_reads + 1")
        if not self.noise_levels or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in self.noise_levels
        ):
            raise ValueError("noise_levels must contain finite nonnegative numbers")
        if tuple(sorted(set(self.noise_levels))) != self.noise_levels:
            raise ValueError("noise_levels must be strictly increasing")
        if not any(self.noise_levels):
            raise ValueError("include a positive noise level for the budget experiment")
