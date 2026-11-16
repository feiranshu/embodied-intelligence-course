from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Frame:
    image: Any
    sequence: int
    captured_at: float


@dataclass
class Candidate:
    id: str
    label: str
    color: str
    shape: str
    bbox: tuple[float, float, float, float]
    marker_id: int | None = None

    @property
    def center(self):
        x, y, w, h = self.bbox
        return x + w / 2, y + h / 2

    def to_dict(self):
        return asdict(self)


@dataclass
class Selection:
    intent: str = "find"
    candidate_ids: list[str] = field(default_factory=list)
    question: str = ""
    reply: str = ""
