"""configs.py -- the placement configurations step_4b_refine.py runs and step_8 scores.

A configuration is one way of turning MotionBERT's pass-1 rotations into a placed skeleton:

    shape        'motionbert'  step_2b's per-camera betas, mesh rescaled to the stature (step_3)
                 'adult'       10 betas fitted to the subject's bone lengths (step_2c), metric
                 'kid'         the same with AGORA's kid template as an 11th direction (step_2c)
                 'cohort_kid'  the cohort-median kid shape (step_2c --cohort): the deployable version
    placement    'pnp'         step_4's rigid PnP per frame + RTS root smoothing; no refinement
                 'ray_const'   per-frame ray rotation (MotionBERT orientation + off-axis correction)
                               + one constant per trial from PnP; translation solved
                 'ray'         the same without the constant
                 'stance'      one rotation per trial (PnP over the first 1.5 s); translation solved
                 'free'        rotation and translation solved per frame (temporally smoothed PnP)
    body         True          the 23 joint angles are refined against the 2D; False keeps MotionBERT's
    exclude_2d   H36M joints left out of the reprojection term (PnP: its correspondences)
    w_foot       grounded ankle keeps a constant horizontal lab position (m^-2; 0 = off)
    w_floor      hinge: no ankle below the lab floor (0 = off)
    w_floor_contact  grounded ankle held AT floor_contact_m above the floor, two-sided (0 = off)
    floor_contact_m  ankle-joint height above the floor while grounded

Contact detection is in seconds / px per second so the same numbers serve 30 and 60 fps.
Everything else (robust loss, pull and smoothness weights) is fixed at the values settled on
BioCV User03 and Korea B023 -- see the sample_* experiment scripts' history.

The list is a CHAIN from the original pipeline to the child recipe, one change per step, then a
LEAVE-ONE-OUT set from the child recipe, then two references.  'all' names every configuration;
run a subset with --configs C0,C8.
"""
HIPS = 'Hip,RHip,LHip'
HEAD = 'Spine,Nose,Head'
FLOOR = dict(w_foot=3000.0, w_floor=3000.0)
CONTACT = dict(w_floor_contact=3000.0, floor_contact_m=0.025)

DEFAULTS = dict(shape='motionbert', placement='ray_const', body=True, exclude_2d='',
                w_foot=0.0, w_floor=0.0, w_floor_contact=0.0, floor_contact_m=0.025,
                w_pull=30.0, w_smooth=30.0, w_trans_acc=1e4, sigma_px=5.0, huber=20.0,
                conf_thresh=0.3, stance_sec=1.5,
                min_contact_s=0.1, foot_speed_px_s=90.0, foot_lowest_margin_px=30.0,
                iters_global=60, iters=300)

CHAIN = {
    # the original pipeline and its hip fix
    'C0': dict(shape='motionbert', placement='pnp', body=False, exclude_2d='',
               note='original: MotionBERT shape, PnP on every joint'),
    'C1': dict(shape='motionbert', placement='pnp', body=False, exclude_2d=HIPS,
               note='PnP without the hips'),
    # placement geometry, MotionBERT's pose kept
    'C2': dict(shape='motionbert', placement='ray_const', body=False, exclude_2d=HIPS,
               note='frozen ray rotation, translation solved, MotionBERT pose kept'),
    # the articulated refinement and its constraints
    'C3': dict(shape='motionbert', placement='ray_const', body=True, exclude_2d=HIPS,
               note='+ body angles refined against the 2D'),
    'C4': dict(shape='motionbert', placement='ray_const', body=True, exclude_2d=HIPS, **FLOOR,
               note='+ feet pinned while grounded, floor hinge'),
    'C5': dict(shape='motionbert', placement='ray_const', body=True, exclude_2d=HIPS, **FLOOR, **CONTACT,
               note='+ grounded ankle held at its floor height'),
    # shape
    'C6': dict(shape='adult', placement='ray_const', body=True, exclude_2d=HIPS + ',' + HEAD, **FLOOR, **CONTACT,
               note='adult SMPL fitted to bone lengths (head/spine out of the 2D term)'),
    'C7': dict(shape='kid', placement='ray_const', body=True, exclude_2d=HIPS + ',' + HEAD, **FLOOR, **CONTACT,
               note='kid-blend shape fitted to bone lengths'),
    'C8': dict(shape='kid', placement='ray_const', body=True, exclude_2d=HEAD, **FLOOR, **CONTACT,
               note='CHILD RECIPE: kid shape, hips back in the 2D term'),
    'C9': dict(shape='cohort_kid', placement='ray_const', body=True, exclude_2d=HEAD, **FLOOR, **CONTACT,
               note='child recipe with the cohort-median kid shape (deployable)'),
}

# one component removed from the child recipe at a time
LEAVE_ONE_OUT = {
    'L1': dict(CHAIN['C8'], shape='motionbert', note='recipe minus kid shape'),
    'L2': dict(CHAIN['C8'], body=False, note='recipe minus body refinement'),
    'L3': dict(CHAIN['C8'], exclude_2d='', note='recipe minus the head/spine exclusion'),
    'L4': dict(CHAIN['C8'], w_foot=0.0, w_floor_contact=0.0, note='recipe minus foot pin and floor contact'),
    'L5': dict(CHAIN['C8'], w_floor_contact=0.0, note='recipe minus floor contact'),
    'L6': dict(CHAIN['C8'], placement='free', note='recipe with free per-frame rotation'),
}

REFERENCE = {
    'R1': dict(CHAIN['C8'], placement='pnp', body=False, exclude_2d=HIPS, note='kid shape through rigid PnP'),
}

CONFIGS = {}
for _group in (CHAIN, LEAVE_ONE_OUT, REFERENCE):
    for _k, _v in _group.items():
        CONFIGS[_k] = dict(DEFAULTS, **_v)

ORDER = list(CONFIGS)


def select(spec):
    """'all' | 'chain' | 'loo' | comma-separated ids -> [(id, config)]."""
    if spec in (None, '', 'all'):
        ids = ORDER
    elif spec == 'chain':
        ids = list(CHAIN)
    elif spec == 'loo':
        ids = list(LEAVE_ONE_OUT)
    else:
        ids = [s.strip() for s in spec.split(',') if s.strip()]
        bad = [i for i in ids if i not in CONFIGS]
        if bad:
            raise SystemExit(f'unknown configuration(s) {bad}; known: {", ".join(ORDER)}')
    return [(i, CONFIGS[i]) for i in ids]


def legend():
    return '\n'.join(f'{i:4s} {c["shape"]:11s} {c["placement"]:10s} body={"on " if c["body"] else "off"} '
                     f'2D-{c["exclude_2d"] or "all":32s} foot={c["w_foot"]:.0f} floor={c["w_floor"]:.0f} '
                     f'contact={c["w_floor_contact"]:.0f}  {c["note"]}' for i, c in CONFIGS.items())


if __name__ == '__main__':
    print(legend())
