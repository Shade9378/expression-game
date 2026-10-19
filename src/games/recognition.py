from __future__ import annotations

import random
from dataclasses import dataclass

from constants import CLASSES
from games.catalog import Reference, ReferenceCatalog


@dataclass(frozen=True)
class RecognitionResult:
    reference_id: str
    target: str
    answer: str | None
    outcome: str
    points: int
    elapsed: float


class RecognitionGame:
    def __init__(
        self,
        catalog: ReferenceCatalog,
        level: int = 0,
        rounds: int = 7,
        seed: int = 42,
        timed: bool = False,
    ) -> None:
        if level not in range(3) or not 1 <= rounds <= 30:
            raise ValueError("Invalid recognition game settings")
        self.catalog, self.level, self.rounds = catalog, level, rounds
        self.rng = random.Random(seed)
        self.timed = timed
        self.results: list[RecognitionResult] = []
        self.reference: Reference | None = None
        self.choices: tuple[str, ...] = ()
        self.started_at = 0.0
        self.state = "idle"
        self.used: set[str] = set()
        self.labels: list[str] = []

    @property
    def score(self) -> int:
        return sum(result.points for result in self.results)

    @property
    def time_limit(self) -> float:
        return (20.0, 15.0, 10.0)[self.level]

    def next_round(self, now: float) -> Reference | None:
        if self.state == "playing":
            raise ValueError("Answer or skip the current round first")
        if len(self.results) >= self.rounds:
            self.state = "finished"
            return None
        if not self.labels:
            self.labels = list(self.catalog.labels)
            self.rng.shuffle(self.labels)
        target = self.labels.pop()
        reference = self.catalog.choose(target, self.level, self.rng, self.used)
        self.used.add(reference.sample_id)
        choices = self.rng.sample([c for c in CLASSES if c != target], 3)
        choices.append(target)
        self.rng.shuffle(choices)
        self.reference, self.choices = reference, tuple(choices)
        self.started_at, self.state = now, "playing"
        return reference

    def answer(self, answer: str | None, now: float) -> RecognitionResult:
        if self.state != "playing" or self.reference is None:
            raise ValueError("No unanswered recognition round")
        if answer is not None and answer not in self.choices:
            raise ValueError("Answer is not a displayed choice")
        elapsed = max(0.0, now - self.started_at)
        outcome = "correct" if answer == self.reference.label else "wrong"
        if answer is None:
            outcome = "skipped"
        if self.timed and elapsed >= self.time_limit:
            outcome = "timeout"
        points = 100 * (self.level + 1) if outcome == "correct" else 0
        if points and self.timed:
            points += round(25 * max(0.0, 1 - elapsed / self.time_limit))
        result = RecognitionResult(
            self.reference.sample_id, self.reference.label, answer, outcome, points, elapsed
        )
        self.results.append(result)
        self.state = "answered"
        return result

    def tick(self, now: float) -> RecognitionResult | None:
        if self.state == "playing" and self.timed and now - self.started_at >= self.time_limit:
            return self.answer(None, now)
        return None
