#!/usr/bin/env python3
"""Copy a trial's Analysis/ outputs from the working tree to RESULTS_DIR.

On Colab the working tree lives on the VM's local disk, which is destroyed when
the runtime ends -- so this needs running after each trial, not once at the end
of a session.

Only Analysis/ is copied: every step writes its outputs there, so it is the
whole output surface of the pipeline. Videos are skipped by default because the
step_5 / step_7 renders dwarf everything else; the npz results are a few MB.

    python3 tools/sync_results.py --user User28 --action P28_CMJM_01
    python3 tools/sync_results.py --user User28              # every action
    python3 tools/sync_results.py --all                      # every trial
    python3 tools/sync_results.py --all --include-videos
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRIAL_DIR, RESULTS_DIR

VIDEO_EXT = ('.mp4', '.avi', '.mov')


def _is_stale(src, dst):
    """True if dst is missing or looks older/different than src.

    Cheap stand-in for rsync: mtime + size is enough here because outputs are
    only ever rewritten wholesale by a step, never appended to.
    """
    if not os.path.exists(dst):
        return True
    s, d = os.stat(src), os.stat(dst)
    return s.st_size != d.st_size or s.st_mtime > d.st_mtime + 1


def sync_trial(user, action, include_videos=False, dry_run=False):
    src_root = os.path.join(TRIAL_DIR, user, action, 'Analysis')
    dst_root = os.path.join(RESULTS_DIR, user, action, 'Analysis')

    if not os.path.isdir(src_root):
        print(f'  {user}/{action}: no Analysis/ -- skipped')
        return 0, 0

    copied = skipped = 0
    for dirpath, _, filenames in os.walk(src_root):
        rel = os.path.relpath(dirpath, src_root)
        for name in sorted(filenames):
            if not include_videos and name.lower().endswith(VIDEO_EXT):
                continue
            src = os.path.join(dirpath, name)
            dst = os.path.normpath(os.path.join(dst_root, rel, name))
            if not _is_stale(src, dst):
                skipped += 1
                continue
            if not dry_run:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)     # copy2 preserves mtime, so the
            copied += 1                    # staleness check works next run

    verb = 'would copy' if dry_run else 'copied'
    print(f'  {user}/{action}: {verb} {copied}, up-to-date {skipped}')
    return copied, skipped


def _subdirs(path):
    if not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path)
                  if os.path.isdir(os.path.join(path, d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user')
    ap.add_argument('--action')
    ap.add_argument('--all', action='store_true',
                    help='every user/action found under TRIAL_DIR')
    ap.add_argument('--include-videos', action='store_true',
                    help='also copy the step_5 / step_7 renders')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if not args.all and not args.user:
        ap.error('give --user (optionally with --action), or --all')

    users = _subdirs(TRIAL_DIR) if args.all else [args.user]

    print(f'from {TRIAL_DIR}')
    print(f'to   {RESULTS_DIR}')
    total = 0
    for user in users:
        actions = ([args.action] if args.action
                   else _subdirs(os.path.join(TRIAL_DIR, user)))
        for action in actions:
            copied, _ = sync_trial(user, action,
                                   args.include_videos, args.dry_run)
            total += copied
    print(f'{"would copy" if args.dry_run else "copied"} {total} file(s)')


if __name__ == '__main__':
    main()
