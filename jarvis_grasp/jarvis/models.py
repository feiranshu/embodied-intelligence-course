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
    angle_deg: float = 0.0
    area_px: float = 0.0
    confidence: float = 1.0
    position_3d_mm: tuple[float, float, float] | None = None

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2, y + h / 2

    @property
    def size(self) -> tuple[float, float]:
        return self.bbox[2], self.bbox[3]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Selection:
    intent: str = "find"
    candidate_ids: list[str] = field(default_factory=list)
    question: str = ""
    reply: str = ""
    place_direction: str | None = None


@dataclass
class GraspPlan:
    target_id: str
    target_label: str
    center_norm: tuple[float, float]
    angle_deg: float
    bbox_norm: tuple[float, float, float, float]
    camera_to_gripper_mm: tuple[float, float, float]
    pre_grasp_z_mm: float
    grasp_z_mm: float
    lift_z_mm: float
    table_z_mm: float
    gripper_open_raw: int
    gripper_close_raw: int
    gripper_held_range_raw: tuple[int, int]
    estimated_base_xyz_mm: tuple[float, float, float] | None = None
    place_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraspResult:
    success: bool
    stage: str
    target_id: str
    target_label: str
    gripper_raw: int | None = None
    gripper_load: int | None = None
    lifted: bool = False
    verified: bool = False
    reason: str = ""
    telemetry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
