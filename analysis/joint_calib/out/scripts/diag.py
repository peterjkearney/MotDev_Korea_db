import sys, numpy as np, torch, argparse
sys.path.insert(0, 'analysis/joint_calib')
import smpl_op_fit as F
ap = argparse.ArgumentParser(); ap.add_argument('ds'); ap.add_argument('root'); ap.add_argument('subj')
ap.add_argument('--extra', default='')
a = ap.parse_args()
sys.argv = ['x', a.ds, a.root] + a.extra.split()
p = argparse.ArgumentParser()
args = F.__dict__  # reuse defaults by parsing
import types
ns = types.SimpleNamespace(shoulder_bone='spine3', robust_mm=30.0, lam_beta=1.0, lam_torso=100.0, lam_pose=1.0,
                           sigma_offset_mm=100.0, iters=200, two_bone=True, sigma_child_mm=50.0, stature_mm=None, sigma_stature_mm=10.0)
for kv in a.extra.split():
    k, v = kv.split('='); setattr(ns, k, (v == 'True') if isinstance(getattr(ns, k), bool) else float(v))
rng = np.random.default_rng(0)
if a.ds == 'biocv':
    X, nv, src = F.load_biocv(a.root, a.subj, '/Volumes/Expansion/MotorDevelopment/BioCV', 4, 150, rng)
else:
    X, nv, src = F.load_korea(a.root, a.subj, 4, 150, rng)
model, names = F.fit_subject(X, a.ds == 'korea', ns, print)
print(F.report(model, X, names, a.subj, None, {}))
B = F.B
def seg(P, a_, b_): return np.linalg.norm(P[:, a_] - P[:, b_], axis=-1)
with torch.no_grad():
    kp, J24, R = model()
kpn = {k: v.numpy() for k, v in kp.items()}
Mk = np.stack([kpn.get(n, np.full((len(X), 3), np.nan)) for n in F.BODY25_NAMES], 1)
print('segment medians data / model (model uses SMPL-24 limb joints + fitted hip/shoulder kp), mm')
for nm, (x, y) in {'thigh': ('Hip', 'Knee'), 'shank': ('Knee', 'Ankle'), 'uarm': ('Shoulder', 'Elbow'),
                   'farm': ('Elbow', 'Wrist'), 'ankle-heel': ('Ankle', 'Heel'), 'ear-ankle': ('Ear', 'Ankle')}.items():
    d = np.nanmedian(np.r_[seg(X, B['L'+x], B['L'+y]), seg(X, B['R'+x], B['R'+y])]) * 1000
    m = np.nanmedian(np.r_[seg(Mk, B['L'+x], B['L'+y]), seg(Mk, B['R'+x], B['R'+y])]) * 1000
    print(f'  {nm:10s} {d:6.0f} {m:6.0f} {m-d:+5.0f}')
for nm, (x, y) in {'neck-midhip': ('Neck', 'MidHip'), 'nose-midhip': ('Nose', 'MidHip'), 'hipw': ('LHip', 'RHip'), 'shw': ('LShoulder', 'RShoulder')}.items():
    d = np.nanmedian(seg(X, B[x], B[y])) * 1000; m = np.nanmedian(seg(Mk, B[x], B[y])) * 1000
    print(f'  {nm:10s} {d:6.0f} {m:6.0f} {m-d:+5.0f}')
with torch.no_grad():
    v, Jr = model.rest()
print('rest J24 thigh/shank', np.linalg.norm(Jr[1]-Jr[4]).item()*1000, np.linalg.norm(Jr[4]-Jr[7]).item()*1000)
