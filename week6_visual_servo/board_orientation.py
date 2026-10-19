"""Orient a rectangular chessboard from the blue circle outside its marked corner."""
from __future__ import annotations

import cv2
import numpy as np


def sample_board_corners(frame, record, cols, rows):
    """Use the capture-time order; old images can be oriented from a visible blue circle."""
    stored = record.get('ordered_corners')
    if stored is not None:
        points = np.asarray(stored,dtype=np.float32)
        if points.shape not in ((cols*rows,2),(cols*rows,1,2)) or not np.isfinite(points).all():
            raise ValueError(f"样本 {record.get('name','')} 的已保存角点格式错误")
        return points.reshape(-1,1,2).copy()
    from common import chessboard_corners
    corners = chessboard_corners(frame,cols,rows)
    return None if corners is None else orient_board_corners(frame,corners,cols,rows)[0]


def _variants(corners, cols, rows):
    grid = np.asarray(corners, dtype=np.float32).reshape(rows, cols, 1, 2)
    return [grid, grid[:, ::-1], grid[::-1], grid[::-1, ::-1]]


def orient_board_corners(frame, corners, cols, rows, previous=None):
    """Return ordered corners and diagnostics; never add a capture rejection rule."""
    points = np.asarray(corners, dtype=np.float32).reshape(rows, cols, 2)
    cells = np.concatenate([np.linalg.norm(np.diff(points, axis=1),axis=2).ravel(),
                            np.linalg.norm(np.diff(points, axis=0),axis=2).ravel()])
    cell = max(1., float(np.median(cells)))
    ideal = np.mgrid[0:cols,0:rows].T.reshape(-1,2).astype(np.float32)
    H, _ = cv2.findHomography(points.reshape(-1,2), ideal)
    candidates = []
    if H is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,np.array([85,45,25],np.uint8),np.array([140,255,255],np.uint8))
        kernel = max(1,int(round(cell*.08)))
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((kernel,kernel),np.uint8))
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        endpoints = np.array([[0,0],[cols-1,0],[0,rows-1],[cols-1,rows-1]],dtype=float)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not .08*cell*cell <= area <= 25*cell*cell:
                continue
            perimeter = cv2.arcLength(contour,True)
            circularity = 4*np.pi*area/max(perimeter*perimeter,1)
            _,_,width,height = cv2.boundingRect(contour)
            if circularity < .22 or not .25 <= width/max(height,1) <= 4:
                continue
            moments = cv2.moments(contour)
            if moments['m00'] <= 0:
                continue
            center = np.array([moments['m10']/moments['m00'],moments['m01']/moments['m00']],dtype=np.float32)
            local = cv2.perspectiveTransform(center.reshape(1,1,2),H).reshape(2)
            # The physical mark is beyond a corner, not blue handwriting alongside a row.
            outside_x = local[0]<-.35 or local[0]>cols-1+.35
            outside_y = local[1]<-.35 or local[1]>rows-1+.35
            distances = np.linalg.norm(endpoints-local,axis=1)
            nearest = int(np.argmin(distances))
            if not (outside_x and outside_y) or distances[nearest]>7:
                continue
            score = area*circularity/(1+distances[nearest]**2)
            candidates.append((score,nearest,center))
    variants = _variants(corners,cols,rows)
    if candidates:
        _,index,center = max(candidates,key=lambda item:item[0])
        ordered = variants[index].reshape(-1,1,2).copy()
        return ordered, dict(source='blue_circle',marker_found=True,
                             marker_center=center.astype(float).tolist(),
                             message='蓝圈已识别：0固定在蓝圈旁的内角点')
    if previous is not None and np.asarray(previous).shape == np.asarray(corners).shape:
        old = np.asarray(previous).reshape(-1,2)
        old = old-old.mean(axis=0)
        def displacement(candidate):
            current = candidate.reshape(-1,2)
            return float(np.mean(np.sum((current-current.mean(axis=0)-old)**2,axis=1)))
        ordered = min(variants,key=displacement).reshape(-1,1,2).copy()
        return ordered, dict(source='tracked',marker_found=False,marker_center=None,
                             message='蓝圈暂未识别：沿用相邻画面的角点方向，请确认蓝圈可见')
    return np.asarray(corners,dtype=np.float32).copy(), dict(
        source='unconfirmed',marker_found=False,marker_center=None,
        message='尚未识别蓝圈，0的物理方向未确认；请让右下角蓝圈进入画面')
