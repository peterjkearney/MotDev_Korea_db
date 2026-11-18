#!/usr/bin/env python3
"""finetune.py -- fine-tune MotionBERT-Mesh on Korea clips (step 4), mixed with its own H36M / 3DPW data.

Starts from the released checkpoint step_2a uses (FT_MB_release_MB_ft_pw3d).  Each step takes one batch of Korea
clips (korea_data.py) under korea_loss.py's masked loss; every --mix-every steps one batch of MotionBERT's own
training data (H36M, 3DPW) under its original MeshLoss is added, so the model keeps what it knew.  The backbone can
stay frozen for the first --freeze-backbone-epochs while the head adapts; the head's BatchNorm layers stay at the
released statistics unless --bn-train.

Before training and after every epoch, dev is scored (korea_loss.korea_metrics): keypoint error vs the pseudo-GT,
H36M error vs camera c's leave-one-out target (root-relative and Procrustes -- the independent number), orientation
error of observed segments; and a few 3DPW test batches (MPJPE / PA-MPJPE, MotionBERT's own joints) to watch for
forgetting.  Checkpoints keep step_2a's format ({'model': state_dict}); best.bin is the lowest dev LOO error.

    python3 training/finetune.py --train .../FinetuneData/train.npz --dev .../FinetuneData/dev.npz --out <dir>
    python3 training/finetune.py ... --smoke             # a few CPU batches, no mixing: does it run and learn

MotionBERT's own data (mesh_det_h36m.pkl, mesh_det_pw3d.pkl) must sit in --data-root next to the SMPL files
(MB_DIR/data/mesh by default), where its readers look for both.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'training'))
sys.path.insert(0, os.path.join(_ROOT, 'pipeline'))
from config import MB_DIR  # noqa: E402
sys.path.insert(0, MB_DIR)
from lib.utils.tools import get_config  # noqa: E402
from lib.utils.learning import load_backbone  # noqa: E402
from lib.model.model_mesh import MeshRegressor  # noqa: E402
from lib.model.loss_mesh import MeshLoss  # noqa: E402
from lib.model.loss import p_mpjpe  # noqa: E402

from kid_body import KidBody  # noqa: E402
from korea_data import KoreaClips  # noqa: E402
from korea_loss import korea_loss, korea_metrics  # noqa: E402

CFG = os.path.join(MB_DIR, 'configs', 'mesh', 'MB_ft_pw3d.yaml')
CKPT = os.path.join(MB_DIR, 'checkpoint', 'mesh', 'FT_MB_release_MB_ft_pw3d', 'best_epoch.bin')
# Korea dev scoring: the targets' eval joints (Nose, Head out -- build_child_gt's eval mask)
EVAL_JOINTS = torch.tensor([i not in (9, 10) for i in range(17)])


def to(dev, gt):
    return {k: v.to(dev) for k, v in gt.items()}


def mb_loss(criterion, output, gt, a):
    d = criterion(output, gt)
    return (a.lambda_3d * d['loss_3d_pos'] + a.lambda_scale * d['loss_3d_scale'] + a.lambda_3dv * d['loss_3d_velocity']
            + a.lambda_lv * d['loss_lv'] + a.lambda_lg * d['loss_lg'] + a.lambda_a * d['loss_a']
            + a.lambda_av * d['loss_av'] + a.lambda_shape * d['loss_shape'] + a.lambda_pose * d['loss_pose']
            + a.lambda_norm * d['loss_norm'])


@torch.no_grad()
def evaluate(model, body, loader, dev, max_batches=None):
    model.eval()
    acc = {}
    for i, (x, gt) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        gt = to(dev, gt)
        m = korea_metrics(model(x.to(dev)), gt, body, EVAL_JOINTS.to(dev))
        for k, (s, n) in m.items():
            a = acc.setdefault(k, [0.0, 0])
            a[0] += s
            a[1] += n
    return {k: s / max(n, 1) for k, (s, n) in acc.items()}


@torch.no_grad()
def evaluate_mix(model, loader, dev, max_batches):
    model.eval()
    e, pa, n = 0.0, 0.0, 0
    for i, (x, gt) in enumerate(loader):
        if i >= max_batches:
            break
        out = model(x.to(dev).float())[0]['kp_3d']
        g = gt['kp_3d'].to(dev).float()
        out, g = out - out[:, :, :1], g - g[:, :, :1]
        e += (out - g).norm(dim=-1).mean().item() * len(x)
        pa += float(np.mean(p_mpjpe(out.reshape(-1, 17, 3).cpu().numpy(), g.reshape(-1, 17, 3).cpu().numpy()))) * len(x)
        n += len(x)
    return dict(mpjpe=e / max(n, 1), pa_mpjpe=pa / max(n, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', required=True)
    ap.add_argument('--dev', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', default=CKPT, help='starting weights (step_2a format)')
    ap.add_argument('--data-root', default=os.path.join(MB_DIR, 'data', 'mesh'))
    ap.add_argument('--mix', default='h36m,pw3d', help="MotionBERT's own sets mixed in ('none' for Korea only)")
    ap.add_argument('--mix-every', type=int, default=2, help='one mixed batch every N Korea batches')
    ap.add_argument('--mix-weight', type=float, default=1.0)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--clip-len', type=int, default=16)
    ap.add_argument('--stride', type=int, default=8, help='frames between training clip starts')
    ap.add_argument('--lr-backbone', type=float, default=5e-6, help='MotionBERT fine-tuned with 5e-5')
    ap.add_argument('--lr-head', type=float, default=5e-5, help='MotionBERT fine-tuned with 5e-4')
    ap.add_argument('--freeze-backbone-epochs', type=int, default=1)
    ap.add_argument('--w-rot', type=float, default=1000.0)
    ap.add_argument('--w-dir', type=float, default=1000.0)
    ap.add_argument('--w-pos', type=float, default=0.5)
    ap.add_argument('--w-vel', type=float, default=10.0)
    ap.add_argument('--no-flip', action='store_true')
    ap.add_argument('--bn-train', action='store_true',
                    help="let the head's BatchNorm layers use and update batch statistics (default: frozen at the "
                         "released ones -- smoke test: Korea batches in train-mode BN doubled the keypoint error "
                         "before any update, 131 vs 64 mm)")
    ap.add_argument('--max-proj-px', type=float, default=15.0)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--dev-batches', type=int, default=0, help='0 = all of dev')
    ap.add_argument('--mix-val-batches', type=int, default=20)
    ap.add_argument('--print-every', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smoke', action='store_true', help='CPU-sized check: 6 sequences, 2 epochs of 10 steps (add --mix none for Korea only)')
    opts = ap.parse_args()
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(opts.out, exist_ok=True)

    args = get_config(CFG)
    args.data_root = opts.data_root
    args.clip_len = opts.clip_len
    model = MeshRegressor(args, backbone=load_backbone(args), dim_rep=args.dim_rep, hidden_dim=args.hidden_dim,
                          dropout_ratio=args.dropout)
    ck = torch.load(opts.ckpt, map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in ck['model'].items()}, strict=True)
    model.to(dev)
    body = KidBody().to(dev)

    seqs = None
    if opts.smoke:
        seqs = np.arange(6)
        opts.epochs = 2
    tr = KoreaClips(opts.train, opts.clip_len, opts.stride, flip=not opts.no_flip, max_proj_px=opts.max_proj_px, seqs=seqs)
    dv = KoreaClips(opts.dev, opts.clip_len, opts.clip_len, flip=False, max_proj_px=1e9, min_ok_frac=0.0,
                    seqs=seqs)
    kw = dict(num_workers=opts.workers, pin_memory=dev == 'cuda')
    tl = DataLoader(tr, batch_size=opts.batch_size, shuffle=True, drop_last=True, **kw)
    dl = DataLoader(dv, batch_size=opts.batch_size, shuffle=False, **kw)
    print(f'Korea train: {len(tr.seqs)} sequences ({tr.n_dropped_proj} dropped, projection > {opts.max_proj_px} px), '
          f'{len(tr)} clips, {len(tl)} batches/epoch | dev {len(dv.seqs)} sequences, {len(dv)} clips | {dev}', flush=True)

    mix_iters, mix_val, criterion = {}, None, None
    if opts.mix != 'none':
        from lib.data.dataset_mesh import MotionSMPL
        criterion = MeshLoss(loss_type=args.loss_type, device=dev)
        # The checkpoint carries its own H36M joint regressor, with the legs' left/right rows swapped relative to
        # data/mesh/J_regressor_h36m_correct.npy, which MotionSMPL builds its target joints from: give the datasets
        # the model's, or prediction and target disagree on which leg is which (released model on 3DPW: 155 mm).
        J_model = model.head.smpl.J_regressor_h36m.detach().cpu().clone()

        def mesh_set(split, name):
            ds = MotionSMPL(args, data_split=split, dataset=name)
            ds.smpl.J_regressor_h36m = J_model.clone()
            return ds
        for name in opts.mix.split(','):
            ds = mesh_set('train', name)
            mix_iters[name] = [DataLoader(ds, batch_size=opts.batch_size, shuffle=True, drop_last=True, **kw), None]
            print(f'mix {name}: {len(ds)} clips', flush=True)
        mix_val = DataLoader(mesh_set('test', 'pw3d'), batch_size=opts.batch_size, shuffle=True, **kw)

    def next_mix(name):
        dl_, it = mix_iters[name]
        try:
            return next(it)
        except (StopIteration, TypeError):
            mix_iters[name][1] = iter(dl_)
            return next(mix_iters[name][1])

    opt = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': opts.lr_backbone},
                             {'params': model.head.parameters(), 'lr': opts.lr_head}], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.lr_decay)
    history = []

    def report(epoch, t0):
        m = evaluate(model, body, dl, dev, opts.dev_batches or None)
        if mix_val is not None:
            m.update({f'pw3d_{k}': v for k, v in evaluate_mix(model, mix_val, dev, opts.mix_val_batches).items()})
        m['epoch'], m['minutes'] = epoch, round((time.time() - t0) / 60, 1)
        history.append(m)
        with open(os.path.join(opts.out, 'history.json'), 'w') as f:
            json.dump(history, f, indent=1)
        print(f'== epoch {epoch}: dev keypoints {m["kp"]:.1f} mm | vs LOO {m["loo"]:.1f} mm, PA {m["loo_pa"]:.1f} | '
              f'orientation {m["rot_deg"]:.1f} deg, bone dir {m["dir_deg"]:.1f} deg'
              + (f' | 3DPW MPJPE {m["pw3d_mpjpe"]:.1f} PA {m["pw3d_pa_mpjpe"]:.1f}' if 'pw3d_mpjpe' in m else ''),
              flush=True)
        return m

    t0 = time.time()
    best = report(0, t0)                                   # the released model: the number to beat
    torch.save({'model': model.state_dict(), 'epoch': 0, 'dev': best}, os.path.join(opts.out, 'best.bin'))
    mix_names = list(mix_iters)
    step = 0
    for epoch in range(1, opts.epochs + 1):
        frozen = epoch <= opts.freeze_backbone_epochs
        for p in model.backbone.parameters():
            p.requires_grad = not frozen
        model.train()
        if not opts.bn_train:                              # keep the released normalisation statistics (see --bn-train)
            for mod in model.modules():
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        run = {}
        for i, (x, gt) in enumerate(tl):
            if opts.smoke and i >= 10:
                break
            gt = to(dev, gt)
            L = korea_loss(model(x.to(dev)), gt, body)
            loss = opts.w_rot * L['rot'] + opts.w_dir * L['dir'] + opts.w_pos * L['pos'] + opts.w_vel * L['vel']
            opt.zero_grad()
            loss.backward()
            if mix_names and step % opts.mix_every == 0:
                xm, gm = next_mix(mix_names[(step // opts.mix_every) % len(mix_names)])
                gm = {k: v.to(dev).float() for k, v in gm.items()}
                lm = opts.mix_weight * mb_loss(criterion, model(xm.to(dev).float()), gm, args)
                lm.backward()
                L['mix'] = lm.detach()
            opt.step()
            step += 1
            for k, v in {**L, 'total': loss}.items():
                run[k] = 0.9 * run.get(k, float(v)) + 0.1 * float(v)
            if i % opts.print_every == 0:
                print(f'  epoch {epoch} [{i}/{len(tl)}]{" (backbone frozen)" if frozen else ""} '
                      + ' '.join(f'{k} {v:.4g}' for k, v in run.items()), flush=True)
        sched.step()
        m = report(epoch, t0)
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'dev': m}, os.path.join(opts.out, f'epoch_{epoch}.bin'))
        if m['loo'] < best['loo']:
            best = m
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'dev': m}, os.path.join(opts.out, 'best.bin'))
            print(f'   new best (dev LOO {m["loo"]:.1f} mm) -> best.bin', flush=True)


if __name__ == '__main__':
    main()
