"""Local calibration corpus.

Samples live under `data/corpus/{human,machine}/` as a `.txt` plus a `.json`
sidecar. That directory is gitignored: the human half is expected to be the
user's own writing, and it must never leave the machine or enter version
control.

Calibrating on your own writing is not a compromise, it's the point. A detector
tuned to a generic corpus will mis-score an individual voice; one tuned partly
on your prose learns not to flag the things you actually do.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import settings

LABELS = ("human", "machine")
REGISTERS = ("academic", "professional", "casual", "creative", "unknown")

# Below this, the statistics that matter are too noisy to teach the model
# anything reliable.
MIN_SAMPLE_WORDS = 120


@dataclass(slots=True)
class Sample:
    id: str
    label: str  # "human" | "machine"
    text: str
    register: str = "unknown"
    topic: str = ""
    generator: str = ""  # model id for machine text, source note for human
    notes: str = ""

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def metadata(self) -> dict:
        data = asdict(self)
        data.pop("text")
        data["word_count"] = self.word_count
        return data


def corpus_root() -> Path:
    return settings.data_dir / "corpus"


def _slug(text: str, limit: int = 32) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned[:limit] or "sample"


def make_id(text: str, hint: str = "") -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"{_slug(hint)}-{digest}" if hint else digest


def add(
    text: str,
    label: str,
    *,
    register: str = "unknown",
    topic: str = "",
    generator: str = "",
    notes: str = "",
) -> Sample | None:
    """Store a sample. Returns None if it is too short or already present."""
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}, got {label!r}")

    text = text.strip()
    if len(text.split()) < MIN_SAMPLE_WORDS:
        return None

    sample = Sample(
        id=make_id(text, topic or label),
        label=label,
        text=text,
        register=register,
        topic=topic,
        generator=generator,
        notes=notes,
    )

    directory = corpus_root() / label
    directory.mkdir(parents=True, exist_ok=True)
    text_path = directory / f"{sample.id}.txt"
    if text_path.exists():
        return None  # exact duplicate

    text_path.write_text(sample.text, encoding="utf-8")
    (directory / f"{sample.id}.json").write_text(
        json.dumps(sample.metadata(), indent=2), encoding="utf-8"
    )
    return sample


def load(label: str | None = None) -> list[Sample]:
    """Load every stored sample, optionally filtered by label."""
    samples: list[Sample] = []
    for current in LABELS:
        if label and current != label:
            continue
        directory = corpus_root() / current
        if not directory.exists():
            continue
        for text_path in sorted(directory.glob("*.txt")):
            meta_path = text_path.with_suffix(".json")
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            samples.append(
                Sample(
                    id=text_path.stem,
                    label=current,
                    text=text_path.read_text(encoding="utf-8"),
                    register=meta.get("register", "unknown"),
                    topic=meta.get("topic", ""),
                    generator=meta.get("generator", ""),
                    notes=meta.get("notes", ""),
                )
            )
    return samples


def summary() -> dict:
    """Counts by label, register, and generator — used to warn about the
    imbalances that quietly wreck a calibration run."""
    samples = load()
    by_label: dict[str, int] = {}
    by_register: dict[str, int] = {}
    by_generator: dict[str, int] = {}

    for sample in samples:
        by_label[sample.label] = by_label.get(sample.label, 0) + 1
        key = f"{sample.label}/{sample.register}"
        by_register[key] = by_register.get(key, 0) + 1
        if sample.generator:
            by_generator[sample.generator] = by_generator.get(sample.generator, 0) + 1

    return {
        "total": len(samples),
        "by_label": by_label,
        "by_register": by_register,
        "by_generator": by_generator,
        "total_words": sum(s.word_count for s in samples),
    }


@dataclass(slots=True)
class CorpusWarning:
    severity: str  # "blocker" | "warning"
    message: str


def audit() -> list[CorpusWarning]:
    """Flag the conditions that make a fitted model untrustworthy.

    Reported before training rather than after, because a model trained on a
    lopsided corpus produces confident numbers that are simply wrong, and the
    metrics alone will not reveal it.
    """
    issues: list[CorpusWarning] = []
    samples = load()
    human = [s for s in samples if s.label == "human"]
    machine = [s for s in samples if s.label == "machine"]

    if len(human) < 10 or len(machine) < 10:
        issues.append(
            CorpusWarning(
                "blocker",
                f"Need at least 10 samples per class to fit anything meaningful "
                f"(have {len(human)} human, {len(machine)} machine).",
            )
        )

    if human and machine:
        ratio = max(len(human), len(machine)) / max(min(len(human), len(machine)), 1)
        if ratio > 3:
            issues.append(
                CorpusWarning(
                    "warning",
                    f"Class imbalance {len(human)}:{len(machine)}. The model will "
                    "learn the majority class and the metrics will flatter it.",
                )
            )

    generators = {s.generator for s in machine if s.generator}
    if len(generators) == 1:
        issues.append(
            CorpusWarning(
                "warning",
                f"All machine samples come from one generator ({next(iter(generators))}). "
                "Expect the model to learn that model's signature rather than "
                "machine text in general — accuracy on other models will be lower.",
            )
        )

    if human:
        human_registers = {s.register for s in human}
        machine_registers = {s.register for s in machine}
        only_machine = machine_registers - human_registers
        if only_machine:
            issues.append(
                CorpusWarning(
                    "warning",
                    f"Registers present only in machine text: {sorted(only_machine)}. "
                    "The model can separate the classes on register alone, which "
                    "will not generalize.",
                )
            )

    return issues
