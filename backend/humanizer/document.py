"""Document model.

Every span carries character offsets into `Document.text` so the UI can paint a
heatmap over the original text and the humanizer can splice rewrites back in
without re-parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Sentence:
    text: str
    start: int
    end: int
    para_index: int
    index: int  # global, document-wide

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(slots=True)
class Paragraph:
    text: str
    start: int
    end: int
    index: int
    sentences: list[Sentence] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(slots=True)
class Document:
    text: str
    paragraphs: list[Paragraph] = field(default_factory=list)
    source_format: str = "txt"
    source_name: str = "input"
    meta: dict = field(default_factory=dict)

    @property
    def sentences(self) -> list[Sentence]:
        return [s for p in self.paragraphs for s in p.sentences]

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def slice(self, start: int, end: int) -> str:
        return self.text[start:end]

    def replace_spans(self, edits: list[tuple[int, int, str]]) -> str:
        """Apply (start, end, replacement) edits to the text.

        Edits must not overlap. Applied right-to-left so earlier offsets stay
        valid as we go.
        """
        ordered = sorted(edits, key=lambda e: e[0], reverse=True)
        for i in range(len(ordered) - 1):
            # ordered is descending, so the *next* item ends before this starts
            if ordered[i + 1][1] > ordered[i][0]:
                raise ValueError(
                    f"overlapping edits: {ordered[i + 1][:2]} and {ordered[i][:2]}"
                )
        out = self.text
        for start, end, replacement in ordered:
            out = out[:start] + replacement + out[end:]
        return out
