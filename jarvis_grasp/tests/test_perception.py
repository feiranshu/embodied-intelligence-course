import unittest

import cv2
import numpy as np

from jarvis.models import Frame
from jarvis.perception import ArUcoDetector, ColorShapeDetector, LocalSelector, TemplateTracker


def scene(dx=0, remove_circle=False):
    image = np.full((480, 640, 3), 240, np.uint8)
    if not remove_circle:
        cv2.circle(image, (150 + dx, 180), 28, (0, 0, 230), -1)
    cv2.rectangle(image, (300 + dx, 220), (355 + dx, 275), (0, 0, 230), -1)
    cv2.rectangle(image, (465 + dx, 120), (520 + dx, 175), (230, 50, 0), -1)
    return Frame(image, 1, 0.0)


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        self.detector = ColorShapeDetector()
        self.candidates = self.detector.detect(scene())
        self.selector = LocalSelector()

    def test_real_pixels_produce_distinct_candidates(self):
        self.assertEqual([(item.color, item.shape) for item in self.candidates],
                         [("red", "circle"), ("red", "square"), ("blue", "square")])
        self.assertAlmostEqual(self.candidates[0].center[0], 150 / 640, delta=0.004)

    def test_ambiguous_then_clarification_retains_color(self):
        first = self.selector.select("看看红色那个", self.candidates)
        self.assertEqual(len(first.candidate_ids), 2)
        self.assertIn("圆", first.question)
        second = self.selector.select("方的", self.candidates, [
            {"role": "user", "content": "看看红色那个"},
            {"role": "assistant", "content": first.question}])
        self.assertEqual(second.candidate_ids, [self.candidates[1].id])
        self.assertFalse(second.question)

    def test_negative_answer_excludes_shape(self):
        selected = self.selector.select("不要方的", self.candidates, [
            {"role": "user", "content": "红色的"},
            {"role": "assistant", "content": "圆的还是方的？"}])
        self.assertEqual(selected.candidate_ids, [self.candidates[0].id])

    def test_not_found_is_not_arbitrary_selection(self):
        self.assertEqual(self.selector.select("紫色物体", self.candidates).candidate_ids, [])
        self.assertTrue(self.selector.select("把它看清楚", self.candidates[:1]).question)

    def test_spatial_and_ordinal(self):
        self.assertEqual(self.selector.select("右边那个红色的", self.candidates).candidate_ids,
                         [self.candidates[1].id])
        self.assertEqual(self.selector.select("第二个", self.candidates).candidate_ids,
                         [self.candidates[1].id])
        self.assertEqual(self.selector.select("第十个", self.candidates).candidate_ids, [])

    def test_stop_is_local(self):
        self.assertEqual(self.selector.select("算了，停止", self.candidates).intent, "stop")
        self.assertEqual(self.selector.select("不要动", self.candidates).intent, "stop")

    def test_tracker_follows_image_motion(self):
        tracker = TemplateTracker(scene(), self.candidates[0])
        tracked = tracker.update(scene(dx=30))
        self.assertIsNotNone(tracked)
        self.assertAlmostEqual(tracked.center[0], 180 / 640, delta=0.005)

    def test_tracker_stops_when_target_disappears(self):
        tracker = TemplateTracker(scene(), self.candidates[0])
        self.assertIsNone(tracker.update(scene(remove_circle=True)))

    def test_tracker_rejects_changed_image_geometry(self):
        tracker = TemplateTracker(scene(), self.candidates[0])
        self.assertIsNone(tracker.update(Frame(np.zeros((240, 320, 3), np.uint8), 2, 0)))

    def test_marker_detection_and_tracking(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 7, 80)
        image = np.full((240, 320, 3), 255, np.uint8)
        image[70:150, 100:180] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        frame = Frame(image, 1, 0)
        found = ArUcoDetector().detect(frame)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].marker_id, 7)
        self.assertEqual(self.selector.select("看标记7", found).candidate_ids, ["marker_7"])
        self.assertEqual(TemplateTracker(frame, found[0]).update(frame).marker_id, 7)

    def test_duplicate_marker_ids_cannot_confirm_identity(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.cvtColor(cv2.aruco.generateImageMarker(dictionary, 7, 60), cv2.COLOR_GRAY2BGR)
        image = np.full((240, 320, 3), 255, np.uint8)
        image[70:130, 30:90] = marker
        image[70:130, 180:240] = marker
        with self.assertRaisesRegex(ValueError, "重复"):
            ArUcoDetector().detect(Frame(image, 1, 0))


if __name__ == "__main__":
    unittest.main()
