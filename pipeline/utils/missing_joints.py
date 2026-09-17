"""missing_joints.py -- what MotionBERT is given for a joint the detector did not return.

OpenPose writes a missing joint as (0, 0, 0), and utils/openpose.py makes a
composite joint (Hip, Thorax, Spine, Head) missing when any of its parts is.
MotionBERT's crop_scale leaves such joints out of its bounding box but still
normalises them, so pixel (0, 0) lands in the top-left corner of the crop --
and through the 81-frame window one such joint disturbs up to 40 frames either
side.  Measured on BioCV User03 camera 06 (tools/threshold_sweep.py), that, not
the detector, was most of the gap between OpenPose and YOLO skeletons.

So a missing joint is given the same joint's position from the nearest frame
where it was seen, at a confidence low enough (0.1) that the lifter treats it
as weak and step_4's PnP gate (0.4) never uses it.  Only MotionBERT's input is
filled; the {cam}_2d.npz files stay a record of what was detected.
"""
import numpy as np

FILL_CONF = 0.1


def fill_missing(h2d, conf_fill=FILL_CONF):
    """(T,17,3) x, y, conf -> copy with conf <= 0 joints filled from the nearest seen frame.

    A joint never seen in the clip stays (0, 0, 0).
    """
    out = np.array(h2d, dtype=np.float32, copy=True)
    seen = out[:, :, 2] > 0
    for j in range(out.shape[1]):
        idx = np.where(seen[:, j])[0]
        gaps = np.where(~seen[:, j])[0]
        if len(idx) == 0 or len(gaps) == 0:
            continue
        near = idx[np.abs(idx[None, :] - gaps[:, None]).argmin(axis=1)]
        out[gaps, j, :2] = out[near, j, :2]
        out[gaps, j, 2] = conf_fill
    return out
