"""Blue-marker ordering regression checks without a camera or robot."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np

from board_orientation import board_winding, orient_board_corners, sample_board_corners


class BoardOrientationTests(unittest.TestCase):
    def fixture(self):
        frame=np.full((480,640,3),240,np.uint8)
        grid=np.mgrid[0:8,0:6].T.reshape(6,8,2).astype(np.float32)*40+[100,100]
        cv2.circle(frame,(445,365),14,(255,0,0),4)
        cv2.putText(frame,'BLUE TEXT',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,0,0),2)
        return frame,grid.astype(np.float32)

    def test_all_four_detector_orders_have_same_physical_origin(self):
        frame,grid=self.fixture()
        expected=grid[::-1,::-1].reshape(-1,1,2)
        for order in [grid,grid[:,::-1],grid[::-1],grid[::-1,::-1]]:
            corners,info=orient_board_corners(frame,order.reshape(-1,1,2),8,6)
            self.assertTrue(info['marker_found'])
            np.testing.assert_array_equal(corners,expected)

    def test_camera_rotation_and_perspective_preserve_marked_origin(self):
        frame,grid=self.fixture()
        transforms=[np.array([[-1.,0,639],[0,-1,479],[0,0,1]]),
                    cv2.getPerspectiveTransform(np.float32([[0,0],[639,0],[639,479],[0,479]]),
                                                np.float32([[25,35],[600,5],[630,430],[60,470]]))]
        for H in transforms:
            warped=cv2.warpPerspective(frame,H,(640,480),borderValue=(240,240,240))
            points=cv2.perspectiveTransform(grid.reshape(-1,1,2),H)
            ordered,info=orient_board_corners(warped,points,8,6)
            self.assertTrue(info['marker_found'])
            np.testing.assert_allclose(ordered[0],points[-1],atol=.01)

    def test_marker_absence_is_reported_and_continuity_keeps_order(self):
        frame,grid=self.fixture()
        previous,_=orient_board_corners(frame,grid.reshape(-1,1,2),8,6)
        blank=np.full_like(frame,240)
        corners,info=orient_board_corners(blank,grid.reshape(-1,1,2),8,6,previous)
        np.testing.assert_array_equal(corners,previous)
        self.assertEqual(info['source'],'tracked')
        _,info=orient_board_corners(blank,grid.reshape(-1,1,2),8,6)
        self.assertEqual(info['source'],'unconfirmed')

    def test_tracking_does_not_mirror_when_board_moves(self):
        frame, grid = self.fixture()
        previous, _ = orient_board_corners(frame, grid.reshape(-1, 1, 2), 8, 6)
        centered = grid - grid.mean(axis=(0, 1))
        moved = centered @ np.array([[.20, 1.], [-1., -.20]], dtype=np.float32) + [320, 240]
        blank = np.full_like(frame, 240)
        corners, info = orient_board_corners(blank, moved.reshape(-1, 1, 2), 8, 6, previous)
        self.assertEqual(info['source'], 'tracked')
        self.assertEqual(board_winding(corners, 8, 6), board_winding(previous, 8, 6))

    def test_unrelated_blue_circle_cannot_mirror_an_anchored_board(self):
        frame, grid = self.fixture()
        flat = grid.reshape(-1, 1, 2)
        previous, _ = orient_board_corners(frame, flat, 8, 6)
        false_marker = np.full_like(frame, 240)
        cv2.circle(false_marker, (55, 365), 14, (255, 0, 0), 4)
        unanchored, info = orient_board_corners(false_marker, flat, 8, 6)
        self.assertTrue(info['marker_found'])
        self.assertNotEqual(board_winding(unanchored, 8, 6), board_winding(previous, 8, 6))
        corners, info = orient_board_corners(false_marker, flat, 8, 6, previous, anchored=True)
        self.assertEqual(info['source'], 'tracked')
        np.testing.assert_array_equal(corners, previous)

    def test_unrelated_blue_circle_cannot_turn_a_tracked_board_by_180_degrees(self):
        frame, grid = self.fixture()
        flat = grid.reshape(-1, 1, 2)
        previous, _ = orient_board_corners(frame, flat, 8, 6)
        false_marker = np.full_like(frame, 240)
        cv2.circle(false_marker, (40, 40), 14, (255, 0, 0), 4)
        unanchored, info = orient_board_corners(false_marker, flat, 8, 6)
        self.assertTrue(info['marker_found'])
        self.assertEqual(board_winding(unanchored, 8, 6), board_winding(previous, 8, 6))
        self.assertGreater(np.linalg.norm(unanchored[0] - previous[0]), 100)
        corners, info = orient_board_corners(false_marker, flat, 8, 6, previous, anchored=True)
        self.assertEqual(info['source'], 'tracked')
        np.testing.assert_array_equal(corners, previous)

    def test_solver_uses_stored_order_instead_of_redetecting(self):
        frame,grid=self.fixture()
        stored=grid[::-1,::-1].reshape(-1,2)
        with patch('common.chessboard_corners',side_effect=AssertionError('must not redetect')):
            actual=sample_board_corners(frame,{'ordered_corners':stored.tolist()},8,6)
        np.testing.assert_array_equal(actual.reshape(-1,2),stored)


if __name__=='__main__':unittest.main()
