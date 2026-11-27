#!/usr/bin/env python3
"""viewer.py -- local side-by-side viewer for rep_review.csv (build_review_sheet.py's sheet).

Plays a rep's three OpenPose stick-figure videos side by side, time-aligned with the sync offsets, shows why
the rep was rejected, and saves your verdict + notes straight back into the CSV.

    ~/anaconda3/envs/motEnv/bin/python analysis/rep_review/viewer.py            # opens http://127.0.0.1:8765
    ... viewer.py --csv other.csv --port 8800 --no-browser

Sync offsets: the GT3D build's refined lag where the rep reached GT3D (reps.csv sync_lag), else align_cameras'
(Aligned/sync_status.csv lag12/lag13); camera 1 frame i == camera c frame i - lag1c.  Failed-sync reps show
whatever offsets align_cameras found (toggle them off in the page).
Videos hold one frame per JSON frame (30 fps data) but are variable-frame-rate renders (header 60 fps, frames
1/30 or 1/20 s apart): the page gets each frame's exact time from ffprobe (/api/pts) and steps by data frame;
playback runs each camera at its own average rate and re-seeks any camera that drifts > 2 frames.

Skeleton (skeleton.py): reps that reached a calibrated camera setup are re-triangulated on request -- reprojected
over each video, drawn from the side (floor frame, across camera 1's line of sight, where depth errors show), bones
coloured by length vs the setup median, plus the setup's per-bone CVs behind the session 'long-bone CV' gate.

Saving rewrites the CSV atomically (tmp + rename).  The first save of a run keeps a timestamped .bak copy.
Don't edit the CSV in Numbers/Excel while the viewer is running -- the two would overwrite each other.
Only serves video files under the Korea root, and listens on localhost only.
"""
import argparse, json, os, re, shutil, subprocess, sys, threading, time, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
import pandas as pd

ROOT = os.environ.get('KOREA_ROOT', '/Volumes/Sandisk_SSD/MotorDevelopment/Korea')
HERE = os.path.dirname(os.path.abspath(__file__))
FFPROBE = shutil.which('ffprobe') or '/usr/local/bin/ffprobe'
sys.path.insert(0, HERE)
import skeleton   # noqa: E402  (re-triangulation for the skeleton overlay / side view / bone panel)
SHOW = ['set', 'group', 'stub', 'child', 'task', 'action', 'rep', 'label', 'split', 'stage_failed', 'reasons',
        'setup_reasons', 'bad_cams', 'sync_status', 'sync_detail', 'sync_min_corr', 'sync_overlap', 'sync_max_lag',
        'frames_raw', 'n_frames',
        'n_usable_frames', 'bone_dev', 'reproj_over_session', 'session_bone_cv', 'notes', 'verdict']


def load_lags(sets):
    lags = {}
    for S in sets:
        p = f'{ROOT}/{S}/{skeleton.ALIGNED}/sync_status.csv'
        if os.path.exists(p):
            d = pd.read_csv(p)
            for s, a, b in zip(d.stub, d.lag12, d.lag13):
                if np.isfinite(a) and np.isfinite(b):
                    lags[s] = [0, int(a), int(b), 'align_cameras']
        p = f'{ROOT}/{S}/{skeleton.GT3D}/reps.csv'
        if os.path.exists(p):
            d = pd.read_csv(p, dtype=str)
            for s, l in zip(d.stub, d.sync_lag):
                m = re.findall(r'-?\d+', str(l))
                if len(m) == 2:
                    lags[s] = [0, int(m[0]), int(m[1]), 'GT3D refined']
    return lags


_pts_cache, _pts_lock = {}, threading.Lock()


def frame_times(path):
    """Presentation time (s) of every video frame, in order.  The renders are variable frame rate (frames 1/30 or
    1/20 s apart, ~25 fps on average, though the header says 60), so frame k's time has to be read, not computed."""
    with _pts_lock:
        if path in _pts_cache:
            return _pts_cache[path]
    out = subprocess.run([FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'packet=pts_time',
                          '-of', 'csv=p=0', path], capture_output=True, text=True, timeout=30).stdout
    pts = sorted(float(x) for x in out.split() if x.strip() not in ('', 'N/A'))
    with _pts_lock:
        _pts_cache[path] = pts
    return pts


def all_people(path):
    """Every person OpenPose detected, per JSON frame (= video frame): [[[x/1920, y/1080, conf] x 25] per person].
    The dataset's wireframe renders draw only the first person of each frame; this lets the page draw them all
    (bystanders standing still also show whether a camera moved).  Undetected joints -> null."""
    with open(path) as f:
        d = json.load(f)
    out = []
    for fr in d['data']:
        people = []
        for sk in fr.get('skeleton', []):
            xy, sc = sk.get('pose', []), sk.get('score', [])
            if len(xy) < 50 or len(sc) < 25:
                continue
            people.append([None if (xy[2 * j] == 0 and xy[2 * j + 1] == 0) or sc[j] <= 0 else
                           [round(xy[2 * j] / 1920, 4), round(xy[2 * j + 1] / 1080, 4), round(sc[j], 2)]
                           for j in range(25)])
        out.append(people)
    return out


class State:
    def __init__(self, csv):
        self.csv = csv
        self.df = pd.read_csv(csv, dtype=str, keep_default_na=False)   # strings in, strings out: exact round trip
        for c in ('notes', 'verdict'):
            if c not in self.df.columns:
                self.df[c] = ''
        self.idx = {s: i for i, s in enumerate(self.df.stub)}
        self.lags = load_lags(sorted(self.df['set'].unique()))
        self.lock = threading.Lock()
        self.backed_up = False

    def rows(self):
        out = []
        for r in self.df[SHOW].to_dict('records'):
            r['lag'] = self.lags.get(r['stub'], [0, 0, 0, 'none'])
            out.append(r)
        return out

    def save(self, stub, notes, verdict):
        with self.lock:
            i = self.idx[stub]
            self.df.at[i, 'notes'] = notes
            self.df.at[i, 'verdict'] = verdict
            if not self.backed_up:
                shutil.copy2(self.csv, self.csv + time.strftime('.%Y%m%d-%H%M%S.bak'))
                self.backed_up = True
            tmp = self.csv + '.tmp'
            self.df.to_csv(tmp, index=False)
            os.replace(tmp, self.csv)


def make_handler(state):
    root_real = os.path.realpath(ROOT)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype='application/json'):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == '/':
                with open(os.path.join(HERE, 'viewer.html'), 'rb') as f:
                    return self._send(200, f.read(), 'text/html; charset=utf-8')
            if u.path == '/api/rows':
                return self._send(200, json.dumps(state.rows()).encode())
            m = re.fullmatch(r'/api/(skel|setupstats)/([^/]+)', u.path)
            if m:
                kind, stub = m.groups()
                i = state.idx.get(stub)
                if i is None:
                    return self._send(404, b'{}')
                S = state.df.at[i, 'set']
                try:
                    out = skeleton.rep_skeleton(S, stub) if kind == 'skel' else skeleton.setup_bone_stats(S, stub)
                except Exception as e:                       # a broken rep must not take the viewer down
                    out = {'available': False, 'why': f'error: {type(e).__name__}: {e}'}
                return self._send(200, json.dumps(out).encode())
            m = re.fullmatch(r'/api/people/([^/]+)/([123])', u.path)
            if m:
                stub, cam = m.groups()
                i = state.idx.get(stub)
                if i is None:
                    return self._send(404, b'{}')
                p = os.path.realpath(os.path.join(ROOT, state.df.at[i, 'set'], f'{stub}_{cam}.json'))
                if not p.startswith(root_real + os.sep) or not os.path.isfile(p):
                    return self._send(200, b'[]')
                return self._send(200, json.dumps(all_people(p)).encode())
            m = re.fullmatch(r'/(video|api/pts)/([^/]+)/([123])', u.path)
            if m:
                kind, stub, cam = m.groups()
                i = state.idx.get(stub)
                if i is None:
                    return self._send(404, b'{}')
                p = os.path.realpath(state.df.at[i, f'video_cam{cam}'])
                if not p.startswith(root_real + os.sep) or not os.path.isfile(p):
                    return self._send(404, b'{}')
                if kind == 'api/pts':
                    return self._send(200, json.dumps(frame_times(p)).encode())
                return self._file(p)
            return self._send(404, b'{}')

        def _file(self, p):
            size = os.path.getsize(p)
            rng = self.headers.get('Range')
            start, end = 0, size - 1
            if rng:
                m = re.match(r'bytes=(\d*)-(\d*)', rng)
                if m:
                    if m.group(1):
                        start = int(m.group(1))
                        end = int(m.group(2)) if m.group(2) else size - 1
                    elif m.group(2):
                        start = size - int(m.group(2))
                end = min(end, size - 1)
            n = end - start + 1
            self.send_response(206 if rng else 200)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(n))
            if rng:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            try:
                with open(p, 'rb') as f:
                    f.seek(start)
                    while n > 0:
                        chunk = f.read(min(1 << 16, n))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        n -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if urlparse(self.path).path != '/api/save':
                return self._send(404, b'{}')
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            if body.get('stub') not in state.idx:
                return self._send(400, b'{"error":"unknown stub"}')
            state.save(body['stub'], str(body.get('notes', '')), str(body.get('verdict', '')))
            return self._send(200, b'{"ok":true}')

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=f'{ROOT}/rep_review.csv')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--aligned', default=skeleton.ALIGNED, help='folder under {root}/{set}: synced reps + sync_status.csv')
    ap.add_argument('--gt3d', default=skeleton.GT3D, help='folder under {root}/{set}: the GT3D build to show')
    a = ap.parse_args()
    skeleton.ALIGNED, skeleton.GT3D = a.aligned, a.gt3d     # before State(): load_lags reads them
    state = State(a.csv)
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), make_handler(state))
    url = f'http://127.0.0.1:{a.port}/'
    print(f'{len(state.df)} reps from {a.csv}\nviewer at {url}  (Ctrl-C to stop)', flush=True)
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
