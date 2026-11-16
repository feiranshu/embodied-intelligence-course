from dataclasses import replace
import math
import re

import cv2
import numpy as np

from .models import Candidate, Selection


COLORS = {"red": ("红色", "红"), "blue": ("蓝色", "蓝"),
          "green": ("绿色", "绿"), "yellow": ("黄色", "黄"),
          "purple": ("紫色", "紫"), "black": ("黑色", "黑"),
          "white": ("白色", "白"), "orange": ("橙色", "橙")}
SHAPES = {"circle": ("圆形", "圆", "球"), "square": ("方形", "方", "盒子", "盒", "正方形")}


class ColorShapeDetector:
    """Detect simple saturated tabletop shapes; not a general object recognizer."""

    def __init__(self, min_area=180):
        self.min_area = min_area

    def detect(self, frame):
        image = frame.image
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ranges = {"red": [(0, 10), (170, 179)], "blue": [(95, 130)],
                  "green": [(35, 85)], "yellow": [(18, 34)]}
        found = []
        for color, intervals in ranges.items():
            mask = np.zeros((height, width), np.uint8)
            for low, high in intervals:
                mask |= cv2.inRange(hsv, (low, 90, 65), (high, 255, 255))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if area < self.min_area or perimeter <= 0:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
                    continue
                polygon = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
                circularity = 4 * math.pi * area / perimeter ** 2
                if len(polygon) == 4 and cv2.isContourConvex(polygon) and 0.6 < w / h < 1.7:
                    shape = "square"
                elif circularity > 0.75 and 0.7 < w / h < 1.4:
                    shape = "circle"
                else:
                    continue
                found.append(Candidate("", COLORS[color][0] + SHAPES[shape][0], color, shape,
                                       (x / width, y / height, w / width, h / height)))
        found.sort(key=lambda item: (item.center[0], item.center[1]))
        return [replace(item, id=f"object_{index}") for index, item in enumerate(found, 1)]


class ArUcoDetector:
    def __init__(self, dictionary=None):
        dictionary = cv2.aruco.DICT_4X4_50 if dictionary is None else dictionary
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary))

    def detect(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame.image)
        if ids is None:
            return []
        if len(set(int(value) for value in ids.flatten())) != len(ids):
            raise ValueError("画面内存在重复的 ArUco 编号，无法唯一确认目标身份")
        height, width = frame.image.shape[:2]
        result = []
        for points, marker_id in zip(corners, ids.flatten()):
            x, y, w, h = cv2.boundingRect(points)
            number = int(marker_id)
            result.append(Candidate(f"marker_{number}", f"标记 {number}", "", "marker",
                                    (x / width, y / height, w / width, h / height), number))
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


class LocalSelector:
    """Deterministic Chinese color/shape dialogue for the offline demonstration."""

    def select(self, text, candidates, history=None):
        text = text.strip()
        if re.search(r"停止|停下|急停|取消|算了|别动|不要动", text) or text.lower() in {"停", "stop", "cancel"}:
            return Selection(intent="stop", reply="已停止当前任务。")
        if text in {"你好", "您好", "谢谢", "谢谢你"}:
            return Selection(intent="chat", reply="你好，可以告诉我想找的物体颜色、形状或位置。")
        positive, negative = _attributes(text)
        prior = []
        if history and history[-1].get("role") == "assistant":
            last_answer = history[-1].get("content", "")
            if any(word in last_answer for word in ("？", "?", "请说明", "请描述")):
                prior = [entry.get("content", "") for entry in history if entry.get("role") == "user"]
        for earlier in reversed(prior):
            old_positive, old_negative = _attributes(earlier)
            for kind in positive:
                if not positive[kind]:
                    positive[kind] = old_positive[kind]
                negative[kind] = list(set(negative[kind] + old_negative[kind]))
        for kind in positive:
            positive[kind] = [value for value in positive[kind] if value not in negative[kind]]
        matches = list(candidates)
        recognized = any(positive.values()) or any(negative.values())
        marker = re.search(r"(?:标记|aruco|marker)[\s_]*(\d+)", text, re.I)
        if marker:
            recognized = True
            matches = [item for item in matches if item.marker_id == int(marker[1])]
        for kind in positive:
            matches = [item for item in matches
                       if (not positive[kind] or getattr(item, kind) in positive[kind])
                       and getattr(item, kind) not in negative[kind]]
        ordinal = re.search(r"第\s*([一二三四五六七八九十\d]+)\s*个", text)
        if ordinal:
            recognized = True
            number = ordinal[1]
            number = int(number) if number.isdigit() else "一二三四五六七八九十".find(number) + 1
            ordered = sorted(matches, key=lambda item: (item.center[0], item.center[1]))
            matches = ordered[number - 1:number] if number > 0 else []
        if "左" in text or "右" in text:
            recognized = True
            matches.sort(key=lambda item: item.center[0], reverse="右" in text)
            if len(matches) > 1 and abs(matches[0].center[0] - matches[1].center[0]) < 0.04:
                return self._question(matches)
            matches = matches[:1]
        if not recognized:
            return Selection(candidate_ids=[item.id for item in candidates],
                             question="请描述目标的颜色、圆形或方形，或说明从左到右第几个？")
        if len(matches) > 1:
            return self._question(matches)
        return Selection(candidate_ids=[item.id for item in matches])

    @staticmethod
    def _question(matches):
        shapes = {item.shape for item in matches}
        colors = {item.color for item in matches}
        if len(shapes) > 1 and shapes <= set(SHAPES):
            question = "你要找圆的还是方的？"
        elif len(colors) > 1 and colors <= set(COLORS):
            question = "你要找" + "、".join(COLORS[color][0] for color in sorted(colors)) + "中的哪个？"
        else:
            question = f"有 {len(matches)} 个相似目标，你要从左到右第几个？"
        return Selection(candidate_ids=[item.id for item in matches], question=question)


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
            return next((item for item in self.marker_detector.detect(frame)
                         if item.marker_id == self.target.marker_id), None)
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
        masked[max(0, py - th // 2):py + th // 2 + 1,
               max(0, px - tw // 2):px + tw // 2 + 1] = 1
        second = float(masked.min())
        if second <= self.max_error and second - score < 0.025:
            return None
        self.left, self.top = left + px, top + py
        x, y = self.left + self.offset[0], self.top + self.offset[1]
        self.target = replace(self.target, bbox=(x / width, y / height, *self.target.bbox[2:]))
        return self.target
