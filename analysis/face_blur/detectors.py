"""detectors.py -- the 2D keypoint models, each returning every person in a frame
as (N,17,3) COCO-17 x, y, confidence.

yolo       YOLO26-m pose (the pipeline's step_1 model), single-stage
rtmpose    RTMDet-m person boxes -> RTMPose-m (body8, 256x192), top-down: the
           pose model only ever sees a crop, so a blur can also act through
           the box.  RTMPose's SimCC scores are not probabilities and can
           exceed 1.
mediapipe  BlazePose heavy (model_complexity=2) in video mode with smoothing:
           FreeMoCap's default 2D tracker.  Single-person, and stateful --
           make one per video.  Also keeps its 33 world landmarks (metric,
           hip-centred), its own monocular 3D, as `world`.
"""
import os

import numpy as np

from common import PYTHON_JETSON

YOLO_WEIGHTS = os.path.join(PYTHON_JETSON, 'models', 'yolo26m-pose.pt')
_MIM = os.path.join(os.path.dirname(__import__('mmpose').__file__), '.mim')
RTMDET_CFG = os.path.join(_MIM, 'demo', 'mmdetection_cfg', 'rtmdet_m_640-8xb32_coco-person.py')
RTMDET_CKPT = os.path.expanduser('~/.cache/torch/hub/checkpoints/'
                                 'rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth')
RTMPOSE_CFG = os.path.join(_MIM, 'configs', 'body_2d_keypoint', 'rtmpose', 'body8',
                           'rtmpose-m_8xb256-420e_body8-256x192.py')
RTMPOSE_CKPT = os.path.expanduser('~/.cache/torch/hub/checkpoints/'
                                  'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth')

# MediaPipe's 33 landmarks -> COCO-17
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


class Yolo:
    name = 'yolo'

    def __init__(self, conf=0.3):
        from ultralytics import YOLO
        self.model = YOLO(YOLO_WEIGHTS)
        self.conf = conf

    def __call__(self, frame):
        r = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return np.zeros((0, 17, 3), np.float32)
        return r.keypoints.data.cpu().numpy().astype(np.float32)


class RtmPose:
    name = 'rtmpose'

    def __init__(self, det_thr=0.3):
        from mmdet.apis import init_detector
        from mmpose.apis import init_model
        from mmpose.utils import adapt_mmdet_pipeline
        self.det = init_detector(RTMDET_CFG, RTMDET_CKPT, device='cpu')
        self.det.cfg = adapt_mmdet_pipeline(self.det.cfg)
        self.pose = init_model(RTMPOSE_CFG, RTMPOSE_CKPT, device='cpu')
        self.det_thr = det_thr

    def __call__(self, frame):
        from mmdet.apis import inference_detector
        from mmpose.apis import inference_topdown
        inst = inference_detector(self.det, frame).pred_instances.cpu().numpy()
        keep = (inst.labels == 0) & (inst.scores >= self.det_thr)
        boxes = inst.bboxes[keep]
        if len(boxes) == 0:
            return np.zeros((0, 17, 3), np.float32)
        out = []
        for s in inference_topdown(self.pose, frame, boxes, bbox_format='xyxy'):
            p = s.pred_instances
            out.append(np.concatenate([p.keypoints[0], p.keypoint_scores[0][:, None]], axis=1))
        return np.asarray(out, np.float32)


class MediaPipe:
    name = 'mediapipe'

    def __init__(self):
        import mediapipe as mp
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2,
                                           smooth_landmarks=True, min_detection_confidence=0.5,
                                           min_tracking_confidence=0.5)
        self.world = None

    def __call__(self, frame):
        import cv2
        h, w = frame.shape[:2]
        r = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if r.pose_landmarks is None:
            self.world = None
            return np.zeros((0, 17, 3), np.float32)
        lm = np.array([(p.x * w, p.y * h, p.visibility) for p in r.pose_landmarks.landmark], np.float32)
        self.world = np.array([(p.x, p.y, p.z) for p in r.pose_world_landmarks.landmark], np.float32)
        return lm[MP_TO_COCO][None]

    def close(self):
        self.pose.close()


def make(name):
    return {'yolo': Yolo, 'rtmpose': RtmPose, 'mediapipe': MediaPipe}[name]()
