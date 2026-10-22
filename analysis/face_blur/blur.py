"""blur.py -- the three ways a face gets blurred.

Each anonymiser has two halves, kept apart so a frame's regions can be saved
(for recall) and then drawn: `find(frame) -> regions` and `apply(frame, regions)`.
A region is one row (x1, y1, x2, y2, score) of the area actually blurred,
already padded and clipped to the frame.

haar    the ethics proposal: "faces ... blurred using a medianBlur function
        within a haar cascade classifier (Cuimei et al., 2017; Viola & Jones,
        2001)".  OpenCV's haarcascade_frontalface_default with the usual
        detectMultiScale(scaleFactor=1.1, minNeighbors=5) and cv2.medianBlur
        over the bare detected box.  The proposal names no kernel size; the
        kernel here is the largest odd size <= half the box width, about as
        strong as deface's (which averages over half the box).  Cuimei et
        al.'s three extra classifiers only REJECT detections, so this plain
        cascade is the upper bound on what that method can blur.
deface  `deface` 1.5 with its CLI defaults: CenterFace on the RGB frame,
        threshold 0.2, --mask-scale 1.3, box blur in an ellipse.  Uses deface's
        own CenterFace and draw_det, so it is the tool, not a re-implementation.
oracle  deface's blur over the projected head sphere in every frame: a blur
        that never misses, to separate "blur moves keypoints" from "missed
        faces were left sharp".
"""
import cv2
import numpy as np

HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_KSIZE_FRAC = 0.5

DEFACE_THRESH = 0.2
DEFACE_MASK_SCALE = 1.3

ORACLE_RADIUS_MM = 130.0      # sphere about the head centre (head.npz) -- covers ears to chin


def _clip(boxes, w, h):
    b = np.asarray(boxes, float).reshape(-1, 5)
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w - 1)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h - 1)
    return b[(b[:, 2] - b[:, 0] >= 2) & (b[:, 3] - b[:, 1] >= 2)]


class Haar:
    shape = 'rect'

    def __init__(self):
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.cascade.empty():
            raise RuntimeError('haarcascade_frontalface_default.xml not found in cv2.data')

    def find(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=HAAR_SCALE_FACTOR,
                                              minNeighbors=HAAR_MIN_NEIGHBORS)
        boxes = [(x, y, x + w, y + h, 1.0) for x, y, w, h in np.reshape(faces, (-1, 4))]
        return _clip(boxes, frame.shape[1], frame.shape[0])

    @staticmethod
    def apply(frame, regions):
        for x1, y1, x2, y2, _ in regions.astype(int):
            k = int((x2 - x1) * HAAR_KSIZE_FRAC) // 2 * 2 + 1      # largest odd <= frac * width
            if k >= 3:
                frame[y1:y2, x1:x2] = cv2.medianBlur(np.ascontiguousarray(frame[y1:y2, x1:x2]), k)
        return frame


class Deface:
    shape = 'ellipse'

    def __init__(self):
        from deface.centerface import CenterFace
        self.centerface = CenterFace(in_shape=None, backend='auto')

    def find(self, frame):
        from deface.deface import scale_bb
        dets, _ = self.centerface(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=DEFACE_THRESH)
        boxes = []
        for d in dets:
            x1, y1, x2, y2 = scale_bb(*d[:4].astype(int), DEFACE_MASK_SCALE)
            boxes.append((x1, y1, x2, y2, float(d[4])))
        return _clip(boxes, frame.shape[1], frame.shape[0])

    @staticmethod
    def apply(frame, regions):
        from deface.deface import draw_det
        for i, (x1, y1, x2, y2, s) in enumerate(regions):
            draw_det(frame, s, i, int(x1), int(y1), int(x2), int(y2), replacewith='blur', ellipse=True)
        return frame


class Oracle:
    """Needs the camera and head.npz's per-frame head centre; find() takes the frame index."""
    shape = 'ellipse'

    def __init__(self, camera, head_centre):
        self.cam = camera
        self.centre = head_centre                      # (T,3) lab mm, NaN where unknown

    def find(self, frame, f):
        c = self.centre[f]
        if not np.isfinite(c).all() or self.cam.depth(c) <= 0:
            return np.zeros((0, 5))
        uv = self.cam.project(c)
        r = self.cam.K[0, 0] * ORACLE_RADIUS_MM / self.cam.depth(c)
        return _clip([(uv[0] - r, uv[1] - r, uv[0] + r, uv[1] + r, 1.0)], frame.shape[1], frame.shape[0])

    apply = Deface.apply


def inside(regions, shape, pts):
    """(P,2) px -> (P,) bool, whether each point lies in any blurred region."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    out = np.zeros(len(pts), bool)
    for x1, y1, x2, y2, _ in np.asarray(regions).reshape(-1, 5):
        if shape == 'rect':
            out |= (pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2)
        else:
            cx, cy, ax, ay = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2
            out |= ((pts[:, 0] - cx) / ax) ** 2 + ((pts[:, 1] - cy) / ay) ** 2 <= 1
    return out
