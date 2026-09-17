"""
frame_decimation.py — shared nearest-frame decimation math, used identically
by extract_2d_keypoints.py (video/YOLO side) and load_mocap_gt.py (mocap
side) so both streams get decimated to the exact same source-frame indices.
Real-time sampling (picks the nearest original frame to each target-fps
tick), not interpolation and not naive frame-stride slicing.
"""

import numpy as np


def nearest_frame_indices(n_frames, src_fps, target_fps):
    """
    Returns the sorted, unique source-frame indices (into a 0..n_frames-1
    array sampled at src_fps) closest to each tick of a target_fps clock
    spanning the same duration. If target_fps >= src_fps, returns
    arange(n_frames) unchanged.
    """
    fps = min(target_fps, src_fps)
    duration = n_frames / src_fps
    n_out = max(1, int(np.floor(duration * fps)) + 1)
    target_times = np.arange(n_out) / fps
    idx = np.clip(np.round(target_times * src_fps).astype(int), 0, n_frames - 1)
    return np.unique(idx)
