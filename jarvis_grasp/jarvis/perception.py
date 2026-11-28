from dataclasses import replace
import math
import re

import cv2
import numpy as np

from .models import Candidate, Selection


COLORS = {
    "red": ("红色", "红"),
    "blue": ("蓝色", "蓝"),
    "green": ("绿色", "绿"),
    "yellow": ("黄色", "黄"),
    "purple": ("紫色", "紫"),
    "black": ("黑色", "黑"),
    "white": ("白色", "白"),
    "orange": ("橙色", "橙"),
}
SHAPES = {
    "circle": ("圆形", "圆", "球"),
    "square": ("方形", "方", "盒子", "盒", "正方形", "方块"),
    "rectangle": ("长方形", "长方体", "积木", "木块", "笔袋", "袋子", "文具袋"),
    "triangle": ("三角形", "三角"),
}

_GRASP_PATTERN = re.compile(
    r"抓取|抓起|抓走|抓一下|抓|夹取|夹起|夹住|夹|拿起|拿走|拿一下|拿|捡起|捡|拾取|取走|pick|grasp|grab",
    re.I,
)
_RELEASE_PATTERN = re.compile(
    r"放下|松开|释放|放开|松爪|松手|张开夹爪|丢下|放回去|放回|release|drop",
    re.I,
)
_PLACE_CLAUSE = re.compile(
    r"(?:并|再)?(?:放到|放回|移到|送到|摆到|放在)\s*(左边|左侧|左面|右边|右侧|右面|前面|前方|后面|后方|中间)"
)
_PLACE_MAP = {
    "左边": "left",
    "左侧": "left",
    "左面": "left",
    "右边": "right",
    "右侧": "right",
    "右面": "right",
    "前面": "front",
    "前方": "front",
    "后面": "back",
    "后方": "back",
    "中间": "center",
}


def _oriented_angle(contour, shape: str) -> float:
    if shape == "circle":
        return 0.0
    (_, _), (rw, rh), angle = cv2.minAreaRect(contour)
    if rw < rh:
        angle += 90.0
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return round(float(angle), 1)


class ColorShapeDetector:
    """Detect tabletop colored shapes with bounding box and grasp orientation.

    Set ``real_world=True`` for real indoor camera scenes with lower-saturation fabrics
    (e.g., sage/teal-green pouches) and targets that may partially touch the image border.
    """

    def __init__(self, min_area=180, *, real_world=False, allow_border=False):
        self.min_area = min_area
        self.real_world = bool(real_world)
        self.allow_border = bool(allow_border or real_world)

    def detect(self, frame):
        image = frame.image
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.real_world:
            # Indoor real-camera profiles: exclude brown wood (H~25-33, S<60),
            # matte-black desk trim/shadows (V<=55 with high blue S), and bright white jaws (S<50, V>215)
            specs = {
                "red": ([(0, 10), (170, 179)], 75, 65, 235),
                "orange": ([(11, 17)], 110, 65, 240),
                "yellow": ([(18, 34)], 90, 80, 245),
                "green": ([(40, 83)], 46, 55, 254),
                "blue": ([(103, 130)], 75, 70, 225),
                "purple": ([(131, 165)], 70, 68, 225),
            }
            open_k, close_k = np.ones((5, 5), np.uint8), np.ones((9, 9), np.uint8)
        else:
            specs = {
                "red": ([(0, 10), (170, 179)], 90, 65, 255),
                "orange": ([(11, 17)], 90, 65, 255),
                "yellow": ([(18, 34)], 90, 65, 255),
                "green": ([(35, 85)], 90, 65, 255),
                "blue": ([(95, 130)], 90, 65, 255),
                "purple": ([(131, 165)], 90, 65, 255),
            }
            open_k, close_k = np.ones((3, 3), np.uint8), None
        found = []
        for color, (intervals, s_min, v_min, v_max) in specs.items():
            mask = np.zeros((height, width), np.uint8)
            for low, high in intervals:
                mask |= cv2.inRange(hsv, (low, s_min, v_min), (high, 255, v_max))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
            if close_k is not None:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if area < self.min_area or perimeter <= 0:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if not self.allow_border and (x <= 0 or y <= 0 or x + w >= width or y + h >= height):
                    continue
                polygon = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
                circularity = 4 * math.pi * area / perimeter ** 2
                (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
                rect_area = max(float(rw * rh), 1.0)
                extent = min(1.0, float(area) / rect_area)
                if len(polygon) == 4 and cv2.isContourConvex(polygon) and 0.6 < w / h < 1.7:
                    shape = "square"
                    confidence = max(0.55, min(1.0, extent))
                elif len(polygon) == 4 and cv2.isContourConvex(polygon) and min(rw, rh) > 0:
                    shape = "rectangle"
                    confidence = max(0.55, min(1.0, extent))
                elif len(polygon) == 3 and cv2.isContourConvex(polygon):
                    shape = "triangle"
                    confidence = max(0.55, min(1.0, float(area) / max(float(w * h) * 0.5, 1.0)))
                elif circularity > 0.75 and 0.7 < w / h < 1.4:
                    shape = "circle"
                    confidence = max(0.6, min(1.0, float(circularity)))
                elif self.real_world and extent >= 0.45 and min(rw, rh) > 0:
                    aspect = max(rw, rh) / max(min(rw, rh), 1.0)
                    shape = "square" if aspect <= 1.35 else "rectangle"
                    confidence = max(0.55, min(1.0, extent))
                else:
                    continue
                angle_deg = _oriented_angle(contour, shape)
                found.append(
                    Candidate(
                        "",
                        COLORS[color][0] + SHAPES[shape][0],
                        color,
                        shape,
                        (x / width, y / height, w / width, h / height),
                        angle_deg=angle_deg,
                        area_px=round(float(area), 1),
                        confidence=round(float(confidence), 3),
                    )
                )
        found.sort(key=lambda item: (item.center[0], item.center[1]))
        return [replace(item, id=f"object_{index}") for index, item in enumerate(found, 1)]


class ArUcoDetector:
    def __init__(self, dictionary=None, *, camera_matrix=None, dist_coeffs=None, marker_mm=0.0):
        dictionary = cv2.aruco.DICT_4X4_50 if dictionary is None else dictionary
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary))
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.camera_matrix = None if camera_matrix is None else np.asarray(camera_matrix, dtype=np.float64)
        self.dist_coeffs = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        self.marker_mm = float(marker_mm)

    def _detect_raw(self, image):
        corners, ids, _ = self.detector.detectMarkers(image)
        if ids is not None:
            return corners, ids
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            enhanced = self.clahe.apply(gray)
            corners, ids, _ = self.detector.detectMarkers(enhanced)
        return corners, ids

    def detect(self, frame):
        corners, ids = self._detect_raw(frame.image)
        if ids is None:
            return []
        if len(set(int(value) for value in ids.flatten())) != len(ids):
            raise ValueError("画面内存在重复的 ArUco 编号，无法唯一确认目标身份")
        height, width = frame.image.shape[:2]
        result = []
        for points, marker_id in zip(corners, ids.flatten()):
            quad = np.asarray(points, dtype=np.float64).reshape(4, 2)
            x, y, w, h = cv2.boundingRect(points)
            number = int(marker_id)
            edge = quad[1] - quad[0]
            angle_deg = round(float(math.degrees(math.atan2(edge[1], edge[0]))), 1)
            area_px = round(float(cv2.contourArea(quad.astype(np.float32))), 1)
            position_3d = None
            if self.camera_matrix is not None and self.marker_mm > 0:
                half = self.marker_mm / 2.0
                obj_pts = np.array(
                    [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
                    dtype=np.float64,
                )
                dist = np.zeros((5, 1), dtype=np.float64) if self.dist_coeffs is None else self.dist_coeffs
                try:
                    ok, _, tvec = cv2.solvePnP(
                        obj_pts, quad, self.camera_matrix, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                except cv2.error:
                    ok, tvec = False, None
                if ok and np.isfinite(tvec).all() and float(tvec[2, 0]) > 0:
                    position_3d = tuple(round(float(v), 2) for v in tvec.reshape(3))
            result.append(
                Candidate(
                    f"marker_{number}",
                    f"标记 {number}",
                    "",
                    "marker",
                    (x / width, y / height, w / width, h / height),
                    number,
                    angle_deg=angle_deg,
                    area_px=area_px,
                    confidence=1.0,
                    position_3d_mm=position_3d,
                )
            )
        return sorted(result, key=lambda item: (item.center[0], item.center[1]))


def _attributes(text):
    positive, negative = {"color": [], "shape": []}, {"color": [], "shape": []}
    for kind, vocabulary in (("color", COLORS), ("shape", SHAPES)):
        for key, aliases in vocabulary.items():
            pattern = "|".join(re.escape(word) for word in sorted((*aliases, key), key=len, reverse=True))
            for match in re.finditer(pattern, text.lower()):
                prefix = text[max(0, match.start() - 4):match.start()]
                destination = negative if re.search(r"(?:不是|不要|别要|非|不选|排除|除了)\s*$", prefix) else positive
                if key not in destination[kind]:
                    destination[kind].append(key)
    return positive, negative


def _extract_place_direction(text: str) -> tuple[str, str | None]:
    match = _PLACE_CLAUSE.search(text)
    if not match:
        return text, None
    direction = _PLACE_MAP.get(match.group(1))
    cleaned = (text[:match.start()] + text[match.end():]).strip()
    return cleaned, direction


def _detect_intent(text: str, prior_user_messages: list[str]) -> tuple[str, str | None]:
    _, place_direction = _extract_place_direction(text)
    if _GRASP_PATTERN.search(text):
        return "grasp", place_direction
    if _RELEASE_PATTERN.search(text):
        return "release", place_direction
    for earlier in reversed(prior_user_messages):
        _, earlier_place = _extract_place_direction(earlier)
        if _GRASP_PATTERN.search(earlier):
            return "grasp", place_direction or earlier_place
    return "find", place_direction


class LocalSelector:
    """Deterministic Chinese object recognition, clarification, and grasp dialogue."""

    def select(self, text, candidates, history=None):
        text = text.strip()
        if re.search(r"停止|停下|急停|取消|算了|别动|不要动", text) or text.lower() in {"停", "stop", "cancel"}:
            return Selection(intent="stop", reply="已停止当前任务。")
        if text in {"你好", "您好", "谢谢", "谢谢你"}:
            return Selection(intent="chat", reply="你好，可以告诉我想找或想抓取的物体颜色、形状或位置。")

        prior = []
        if history and history[-1].get("role") == "assistant":
            last_answer = history[-1].get("content", "")
            if any(word in last_answer for word in ("？", "?", "请说明", "请描述")):
                prior = [entry.get("content", "") for entry in history if entry.get("role") == "user"]

        intent, place_direction = _detect_intent(text, prior)
        if intent == "release":
            return Selection(intent="release", reply="正在放下当前物体。", place_direction=place_direction)

        target_text, _ = _extract_place_direction(text)
        positive, negative = _attributes(target_text)
        for earlier in reversed(prior):
            earlier_target, _ = _extract_place_direction(earlier)
            old_positive, old_negative = _attributes(earlier_target)
            for kind in positive:
                if not positive[kind]:
                    positive[kind] = old_positive[kind]
                negative[kind] = list(set(negative[kind] + old_negative[kind]))
        for kind in positive:
            positive[kind] = [value for value in positive[kind] if value not in negative[kind]]

        matches = list(candidates)
        recognized = any(positive.values()) or any(negative.values())
        marker = re.search(r"(?:标记|aruco|marker)[\s_]*(\d+)", target_text, re.I)
        if marker:
            recognized = True
            matches = [item for item in matches if item.marker_id == int(marker[1])]
        for kind in positive:
            matches = [
                item
                for item in matches
                if (not positive[kind] or getattr(item, kind) in positive[kind])
                and getattr(item, kind) not in negative[kind]
            ]
        ordinal = re.search(r"第\s*([一二三四五六七八九十\d]+)\s*个", target_text)
        if ordinal:
            recognized = True
            number = ordinal[1]
            number = int(number) if number.isdigit() else "一二三四五六七八九十".find(number) + 1
            ordered = sorted(matches, key=lambda item: (item.center[0], item.center[1]))
            matches = ordered[number - 1:number] if number > 0 else []
        if "左" in target_text or "右" in target_text:
            recognized = True
            matches.sort(key=lambda item: item.center[0], reverse="右" in target_text)
            if len(matches) > 1 and abs(matches[0].center[0] - matches[1].center[0]) < 0.04:
                return self._question(matches, intent, place_direction)
            matches = matches[:1]

        if not recognized:
            verb = "抓取" if intent == "grasp" else "查找"
            return Selection(
                intent=intent,
                candidate_ids=[item.id for item in candidates],
                question=f"请描述要{verb}的目标颜色、圆形或方形，或说明从左到右第几个？",
                place_direction=place_direction,
            )
        if len(matches) > 1:
            return self._question(matches, intent, place_direction)
        return Selection(
            intent=intent,
            candidate_ids=[item.id for item in matches],
            place_direction=place_direction,
        )

    @staticmethod
    def _question(matches, intent="find", place_direction=None):
        verb = "抓" if intent == "grasp" else "找"
        shapes = {item.shape for item in matches}
        colors = {item.color for item in matches}
        if len(shapes) > 1 and shapes <= set(SHAPES):
            if shapes == {"circle", "square"}:
                question = f"你要{verb}圆的还是方的？"
            else:
                names = "、".join(SHAPES[shape][0] for shape in sorted(shapes))
                question = f"你要{verb}{names}中的哪个？"
        elif len(colors) > 1 and colors <= set(COLORS):
            question = f"你要{verb}" + "、".join(COLORS[color][0] for color in sorted(colors)) + "中的哪个？"
        else:
            question = f"有 {len(matches)} 个相似目标，你要{verb}从左到右第几个？"
        return Selection(
            intent=intent,
            candidate_ids=[item.id for item in matches],
            question=question,
            place_direction=place_direction,
        )


class TemplateTracker:
    """Short-range image matching; keep the original template to avoid adaptive drift."""

    def __init__(self, frame, target, max_shift=0.18, max_error=0.12):
        self.target = target
        self.max_shift = max_shift
        self.max_error = max_error
        self.shape = frame.image.shape
        self.marker_detector = ArUcoDetector() if target.marker_id is not None else None
        height, width = self.shape[:2]
        x, y, w, h = target.bbox
        self.left = max(0, int(x * width) - 5)
        self.top = max(0, int(y * height) - 5)
        right = min(width, math.ceil((x + w) * width) + 5)
        bottom = min(height, math.ceil((y + h) * height) + 5)
        self.offset = (x * width - self.left, y * height - self.top)
        self.template = frame.image[self.top:bottom, self.left:right].copy()
        if self.template.size == 0:
            raise ValueError("目标框没有图像内容")

    def update(self, frame):
        if frame.image.shape != self.shape:
            return None
        if self.marker_detector:
            return next(
                (item for item in self.marker_detector.detect(frame) if item.marker_id == self.target.marker_id),
                None,
            )
        height, width = self.shape[:2]
        th, tw = self.template.shape[:2]
        dx, dy = int(width * self.max_shift), int(height * self.max_shift)
        left, top = max(0, self.left - dx), max(0, self.top - dy)
        right, bottom = min(width, self.left + tw + dx), min(height, self.top + th + dy)
        region = frame.image[top:bottom, left:right]
        if region.shape[0] < th or region.shape[1] < tw:
            return None
        error = cv2.matchTemplate(region, self.template, cv2.TM_SQDIFF_NORMED)
        score, _, location, _ = cv2.minMaxLoc(error)
        if not np.isfinite(score) or score > self.max_error:
            return None
        # A second equally good match outside the first footprint is ambiguous.
        masked = error.copy()
        px, py = location
        masked[max(0, py - th // 2):py + th // 2 + 1, max(0, px - tw // 2):px + tw // 2 + 1] = 1
        second = float(masked.min())
        if second <= self.max_error and second - score < 0.025:
            return None
        self.left, self.top = left + px, top + py
        x, y = self.left + self.offset[0], self.top + self.offset[1]
        self.target = replace(self.target, bbox=(x / width, y / height, *self.target.bbox[2:]))
        return self.target


def draw_detections(image, candidates, *, selected_id=None, center_fraction=0.2, grasp_stage=""):
    """Render non-destructive detection boxes, grasp axis, and alignment zone onto a preview copy."""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    half = center_fraction / 2.0
    cv2.rectangle(
        canvas,
        (int((0.5 - half) * width), int((0.5 - half) * height)),
        (int((0.5 + half) * width), int((0.5 + half) * height)),
        (110, 175, 145),
        1,
    )
    for item in candidates or []:
        if isinstance(item, dict):
            cid = item.get("id", "")
            bbox = item.get("bbox", (0, 0, 0, 0))
            color_name = item.get("color", "")
            shape_name = item.get("shape", "")
            angle_deg = float(item.get("angle_deg", 0.0))
        else:
            cid = item.id
            bbox = item.bbox
            color_name = item.color
            shape_name = item.shape
            angle_deg = float(item.angle_deg)
        x, y, w, h = bbox
        x1, y1 = int(round(x * width)), int(round(y * height))
        x2, y2 = int(round((x + w) * width)), int(round((y + h) * height))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        is_selected = cid == selected_id
        box_color = (45, 175, 80) if is_selected else (175, 110, 50)
        thickness = 2 if is_selected else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)
        cv2.circle(canvas, (cx, cy), 3, box_color, -1)
        if is_selected:
            rad = math.radians(angle_deg)
            span = max(14, int(max(x2 - x1, y2 - y1) * 0.45))
            dx_a, dy_a = int(round(math.cos(rad) * span)), int(round(math.sin(rad) * span))
            cv2.line(canvas, (cx - dx_a, cy - dy_a), (cx + dx_a, cy + dy_a), (30, 220, 220), 2)
        tag = f"{cid} {color_name} {shape_name}".strip()
        cv2.putText(
            canvas,
            tag,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA,
        )
    if grasp_stage:
        cv2.putText(
            canvas,
            f"GRASP: {grasp_stage.upper()}",
            (14, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 95, 165),
            2,
            cv2.LINE_AA,
        )
    return canvas
