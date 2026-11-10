"""Insert the Korea-data section (7 slides) after slide 3 of child_lifting_progress.pptx, in the deck's style.
Edits the deck in place (keeps a .bak copy); renumbers every slide's footer number.

    python add_data_slides.py            # needs python-pptx; clips/posters from assets/render_raw_examples.py,
                                         # sync signals from assets/sync_b010.npz
"""
import shutil
from pathlib import Path

import numpy as np
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).parent
A = HERE / 'assets'
DECK = HERE / 'child_lifting_progress.pptx'
INSERT_AFTER = 3          # new slides follow this (1-based) slide

INK = RGBColor(0x1F, 0x2A, 0x36)
MUTED = RGBColor(0x5B, 0x6B, 0x7A)
PANEL = RGBColor(0xEE, 0xF1, 0xF4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xC9, 0xD3, 0xDD)
CORAL = RGBColor(0xE4, 0x57, 0x2E)
TEAL = RGBColor(0x2A, 0x9D, 0x8F)
BLUE = RGBColor(0x5D, 0xA5, 0xD6)
GREEN = RGBColor(0x5F, 0xB0, 0x5F)
SETC = {'A': RGBColor(0xB4, 0xBF, 0xCB), 'B': RGBColor(0x5D, 0xA5, 0xD6), 'C': RGBColor(0x2C, 0x3A, 0x4A)}
HEAD, BODY = 'Cambria', 'Calibri'
W = 13.333

shutil.copy(DECK, DECK.with_suffix('.pptx.bak'))
prs = Presentation(DECK)
BLANK = prs.slide_layouts[6]


# ---------------------------------------------------------------- helpers (same look as build.py)
def text(slide, x, y, w, h, runs, size=16, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None, para_space=0, bullets=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'):
        setattr(tf, m, 0)
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if para_space and i:
            p.space_before = Pt(para_space)
        if bullets:
            _bullet(p)
        segs = para if isinstance(para, list) else [(para, {})]
        for seg, opts in segs:
            r = p.add_run()
            r.text = seg
            r.font.name = opts.get('font', font)
            r.font.size = Pt(opts.get('size', size))
            r.font.bold = opts.get('bold', bold)
            r.font.color.rgb = opts.get('color', color)
    return tb


def _bullet(p):
    from pptx.oxml.ns import qn
    pPr = p._p.get_or_add_pPr()
    pPr.set('marL', str(Inches(0.25)))
    pPr.set('indent', str(-Inches(0.25)))
    for tag in ('a:buClr', 'a:buChar'):
        pass
    clr = pPr.makeelement(qn('a:buClr'), {})
    srgb = clr.makeelement(qn('a:srgbClr'), {'val': 'E4572E'})
    clr.append(srgb)
    pPr.append(clr)
    pPr.append(pPr.makeelement(qn('a:buChar'), {'char': '•'}))


def title(slide, t, color=INK):
    text(slide, 0.6, 0.45, W - 1.2, 0.9, t, size=32, bold=True, font=HEAD, color=color,
         anchor=MSO_ANCHOR.MIDDLE, name='Title')


def footer(slide):
    text(slide, 0.6, 7.0, 6, 0.3, 'Child 3D lifting · progress update', size=10, color=MUTED, name='Footer')
    text(slide, W - 1.6, 7.0, 1.0, 0.3, '0', size=10, color=MUTED, align=PP_ALIGN.RIGHT, name='Slide number')


def box(slide, x, y, w, h, fill=PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=None, radius=0.06, line=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if name:
        s.name = name
    return s


def shape_text(s, t, size=13, color=WHITE, bold=True):
    tf = s.text_frame
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'):
        setattr(tf, m, Inches(0.04))
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = t
    r.font.size, r.font.bold, r.font.name = Pt(size), bold, BODY
    r.font.color.rgb = color


def badge(slide, x, y, label, fill=CORAL, d=0.45, size=15, name=None):
    s = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL, name=name)
    shape_text(s, label, size=size)
    return s


def line(slide, x1, y1, x2, y2, color=PALE, width=2.0, dash=False, name=None, arrow=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if arrow:
        from pptx.oxml.ns import qn
        ln = c.line._get_or_add_ln()
        ln.append(ln.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'}))
    if name:
        c.name = name
    return c


def picture(slide, path, x, y, w, name, alt):
    p = slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    p.name = name
    p._element.nvPicPr.cNvPr.set('descr', alt)
    return p


def movie(slide, stem, x, y, w, name):
    from PIL import Image
    pw, ph = Image.open(A / f'{stem}.jpg').size
    h = w * ph / pw
    m = slide.shapes.add_movie(str(A / f'{stem}.mp4'), Inches(x), Inches(y), Inches(w), Inches(h),
                               poster_frame_image=str(A / f'{stem}.jpg'), mime_type='video/mp4')
    m.name = name
    return h


def stat(slide, x, y, w, big, label, color=CORAL, name=None, size=32):
    text(slide, x, y, w, 0.65, big, size=size, bold=True, font=HEAD, color=color, name=name)
    text(slide, x, y + 0.65, w, 0.7, label, size=13, color=MUTED, name=(name + ' label') if name else None)


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


def style_chart(ch, title_txt=None):
    ch.font.name = BODY
    ch.font.size = Pt(11)
    ch.font.color.rgb = MUTED
    if title_txt:
        ch.has_title = True
        ch.chart_title.text_frame.text = title_txt
        r = ch.chart_title.text_frame.paragraphs[0].runs[0]
        r.font.size, r.font.bold, r.font.color.rgb = Pt(13), False, INK


def new_slide():
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    return s


new = []

# ================================================================ D1 the dataset
s = new_slide(); new.append(s)
title(s, 'The dataset: 399 children, three age groups, 12 motor tasks')
rows = [('', 'Group A', 'Group B', 'Group C'),
        ('Age', '20–35 months', '36–53 months', '54–71 months'),
        ('Median height (growth charts)', '0.89 m', '1.01 m', '1.10 m'),
        ('Children', '136', '106', '157'),
        ('Reps (each seen by 3 cameras)', '2,388', '1,551', '2,562'),
        ('Reps per child · typical length', '19 · 3.3 s', '15 · 5.3 s', '19 · 3.0 s'),
        ('Rated "Perfect" by the assessor', '54%', '43%', '63%'),
        ('Task 1', 'Climb stairs, feet together', 'Stand on one foot >3 s', 'Stop a rolling ball with the feet'),
        ('Task 2', 'Descend stairs, feet together', 'Hop 2–3 steps on one foot', 'Bounce a ball once'),
        ('Task 3', 'Overhead ball throw', 'Two-footed big jump', 'Jump over a knee-high rope'),
        ('Task 4', 'Stand on one foot for 1 s', 'Catch a big ball against the chest', 'Jump rope once')]
gt = s.shapes.add_table(len(rows), 4, Inches(0.6), Inches(1.45), Inches(W - 1.2), Inches(0.4 * len(rows)))
gt.name = 'Dataset table'
tbl = gt.table
for j, wd in enumerate([3.4, 2.9, 2.9, 2.93]):
    tbl.columns[j].width = Inches(wd)
for i, r in enumerate(rows):
    for j, v in enumerate(r):
        c = tbl.cell(i, j)
        c.fill.solid()
        c.fill.fore_color.rgb = INK if i == 0 else (PANEL if i >= 7 else WHITE)
        c.margin_left = c.margin_right = Inches(0.1)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = c.text_frame
        tf.text = v
        f = tf.paragraphs[0].runs[0].font if v else None
        if f:
            f.name, f.size = BODY, Pt(14)
            f.bold = i == 0 or j == 0
            f.color.rgb = WHITE if i == 0 else (MUTED if j == 0 else INK)
text(s, 0.6, 6.05, W - 1.2, 0.7,
     'Korean children filmed doing gross-motor-skill tasks from a standard assessment. Group A is outside our target '
     'age range and its stair tasks leave the floor, so it is used only to calibrate joint definitions.',
     size=12, color=MUTED, name='Table note')
footer(s)
notes(s, 'Group = age band; each group does its own four tasks, so tasks are never compared across groups. Each '
         'attempt is a rep, filmed by three fixed cameras and rated Perfect, Good or Bad by the assessor. The jumps '
         '(B3, C3) are where flight phases appear. Counts are from the OpenPose files on our copy of the dataset.')

# ================================================================ D2 raw data + good example
s = new_slide(); new.append(s)
title(s, 'The raw data: three cameras, no shared clock, no video')
bl = [[('Three fixed cameras ', {'bold': True}), ('per child, front and 45° either side', {})],
      [('OpenPose 2D keypoints only, ', {'bold': True}), ('30 fps: the video is withheld for anonymity, and there is no calibration object', {})],
      [('No shared clock: ', {'bold': True}), ('each camera started recording at a different moment', {})],
      [('No identity: ', {'bold': True}), ('OpenPose detects every person in every frame, in no fixed order', {})]]
text(s, 0.6, 1.55, 7.2, 2.5, bl, size=16, para_space=10, bullets=True, name='Raw data bullets')
# camera layout: child in the middle, cameras on a front arc
cx, cy = 10.45, 2.05
kid = box(s, cx - 0.32, cy - 0.32, 0.64, 0.64, fill=CORAL, shape=MSO_SHAPE.OVAL, name='Child')
shape_text(kid, 'child', size=11)
for k, (ang, lab) in enumerate(((0, '1 front'), (-45, '2 right 45°'), (45, '3 left 45°'))):
    a = np.radians(ang)
    px, py = cx + 1.75 * np.sin(a), cy + 1.75 * np.cos(a) * 0.95
    line(s, cx, cy, px, py, color=PALE, width=1.5, dash=True, name=f'Sight line {k+1}')
    cam = box(s, px - 0.55, py - 0.2, 1.1, 0.4, fill=INK, name=f'Camera {k+1}')
    shape_text(cam, lab, size=11)
kid._element.getparent().remove(kid._element); s.shapes._spTree.append(kid._element)   # child on top
vh = movie(s, 'raw_good_B010', 0.6, 4.15, W - 1.2, 'Good example video')
text(s, 0.6, 4.15 + vh + 0.08, W - 1.2, 0.3,
     'A good rep (B010, task 1): the same movement in all three cameras after syncing. Blue = left side, red = right.',
     size=12, color=MUTED, name='Video caption')
footer(s)
notes(s, 'Everything downstream has to be recovered from these keypoints: when each camera started, where the '
         'cameras are, and which detected person is the child. Click the video to play.')

# ================================================================ D3 what goes wrong
s = new_slide(); new.append(s)
title(s, 'What goes wrong in the raw data')
vw = 7.3
h1 = movie(s, 'raw_switch_B237', 0.6, 1.5, vw, 'Wrong-person video')
h2 = movie(s, 'raw_handheld_B286', 0.6, 1.5 + h1 + 0.45, vw, 'Handheld video')
tx = 0.6 + vw + 0.45
tw = W - 0.6 - tx
y2 = 1.5 + h1 + 0.45
for y, head, body, fix in (
        (1.5, 'The wrong person',
         'Camera 3 lists an adult first in 163 of 217 frames, switching 44 times (B237).',
         'Fix: link detections into tracks and keep the person who moves the same way in all three cameras.'),
        (y2, 'A handheld camera',
         'Camera 1 moves every frame (B286), so no single camera position fits: bone lengths vary by 42% '
         '(4–5% in a normal session).',
         'Fix: the session fails the bone-stability check and is excluded.')):
    text(s, tx, y, tw, 0.4, head, size=20, bold=True, font=HEAD, name=f'{head} heading')
    text(s, tx, y + 0.5, tw, 1.0, body, size=14, name=f'{head} text')
    text(s, tx, y + 1.45, tw, 0.8, fix, size=14, color=TEAL, name=f'{head} fix')
footer(s)
notes(s, 'Top: grey is everyone OpenPose detected; red is whoever it listed first, which is what the original '
         'merge used; green is the child after tracking. Bottom: the yellow trail is the mid-hip over the last two '
         'seconds. The tripod camera traces the child\'s real movement; the handheld one adds constant shake. '
         'You spotted B286 by eye; it also shows as weak sync (0.45-0.60 correlation vs 0.9+) and failed calibration.')

# ================================================================ D4 sync
s = new_slide(); new.append(s)
title(s, 'Step 1: put the three cameras on one clock')
steps = [('Find the child', 'Link each camera\'s detections into tracks; keep the set of three that moves together. '
                            'Reps with switches: sync failures 73% → 6%'),
         ('One motion signal', 'The body\'s up/down movement in the image, which looks the same from every camera'),
         ('Match and check', 'Cross-correlate each pair; accept only if all pairs correlate and the three offsets '
                             'agree within 1 frame')]
for i, (h, d) in enumerate(steps):
    y = 1.55 + i * 1.3
    badge(s, 0.6, y, str(i + 1), name=f'Sync badge {i+1}')
    text(s, 1.25, y + 0.02, 4.6, 0.4, h, size=17, bold=True, name=f'Sync step {i+1}')
    text(s, 1.25, y + 0.42, 4.6, 0.85, d, size=13, color=MUTED, name=f'Sync step {i+1} text')
stat(s, 0.6, 5.5, 5.2, '5,080 of 6,501 reps synced', 'A 1,797 · B 1,456 · C 1,827', name='Sync stat', size=24)
z = np.load(A / 'sync_b010.npz')
lag = {1: 0, 2: int(z['lag12']), 3: int(z['lag13'])}
camcol = {1: INK, 2: CORAL, 3: BLUE}
for k, (ttl, shifted, y) in enumerate((('Before: each camera on its own clock', False, 1.45),
                                       (f'After: shifted by the fitted offsets ({lag[2]:+d} and {lag[3]:+d} frames)', True, 4.1))):
    cd = XyChartData()
    for c in (1, 2, 3):
        sig, ok = z[f's{c}'], z[f'v{c}'].astype(bool)
        sig = np.convolve(np.where(ok, sig, 0), np.ones(3) / 3, 'same')
        ser = cd.add_series(f'camera {c}')
        for i in range(len(sig)):
            t = (i + lag[c]) if shifted else i                 # camera-c frame i == camera-1 frame i + lag1c
            ser.add_data_point(t / 30.0, float(sig[i]))
    gf = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, Inches(6.3), Inches(y), Inches(6.45),
                            Inches(2.5), cd)
    gf.name = 'Sync chart ' + ('after' if shifted else 'before')
    ch = gf.chart
    style_chart(ch, ttl)
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.RIGHT
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(11)
    for c, ser in zip((1, 2, 3), ch.plots[0].series):
        ser.format.line.color.rgb = camcol[c]
        ser.format.line.width = Pt(1.75)
        ser.smooth = False
    va, ca = ch.value_axis, ch.category_axis
    va.visible = False
    va.has_major_gridlines = False
    ca.has_major_gridlines = False
    ca.minimum_scale, ca.maximum_scale = 0, 8
    ca.major_unit = 2
    ca.tick_labels.font.size = Pt(10)
    ca.format.line.color.rgb = PALE
    ca.has_title = True
    ca.axis_title.text_frame.text = 'seconds'
    ca.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
footer(s)
notes(s, 'Charts: B010 task 1. Each line is a camera\'s vertical-motion signal (confidence-weighted signed '
         'vertical velocity, 3-frame smooth). Before: the same jumps arrive at different times. After shifting by '
         'the fitted offsets they line up. The vertical signal synced more reps than the original speed signal '
         '(1,343 vs 1,177 of 1,551 in set B before tracking was added). Offsets are whole frames; step 2 refines them.')

# ================================================================ D5 calibration
s = new_slide(); new.append(s)
title(s, 'Step 2: the child is the calibration object')
bl = [[('Solve everything together: ', {'bold': True}),
       ('focal length, lens distortion, camera positions and the child\'s 3D joints', {})],
      [('Constraints: ', {'bold': True}), ('each bone keeps one length in every frame; left and right match', {})],
      [('Refine the sync: ', {'bold': True}),
       ('the first sync is within a frame but exact only about half the time, and mistimed frames bend the calibration', {})],
      [('Floor: ', {'bold': True}), ('fitted through the toes and heels, giving gravity, camera heights and tilts', {})]]
text(s, 0.6, 1.55, 6.3, 3.3, bl, size=16, para_space=12, bullets=True, name='Calibration bullets')
# cycle: calibrate -> refine sync -> recalibrate
cyc = [('Calibrate cameras', 7.6, 1.7), ('Refine sync\n(±4 frames per camera)', 9.9, 3.25), ('Recalibrate,\nthen refine once more', 7.6, 4.8)]
for i, (lab, x, y) in enumerate(cyc):
    b = box(s, x, y, 2.6, 0.95, fill=INK if i != 1 else TEAL, name=f'Cycle {i+1}')
    shape_text(b, lab, size=14)
line(s, 10.2, 2.17, 11.2, 3.2, color=MUTED, width=2, name='Cycle arrow 1', arrow=True)
line(s, 11.2, 4.25, 10.2, 5.25, color=MUTED, width=2, name='Cycle arrow 2', arrow=True)
for nm, (h, d) in enumerate((('Moved camera', 'Reps one camera can\'t explain → separate camera setups (B237: camera 2 moved between tasks 3 and 4)'),
                             ('Second person', 'Several candidate tracks → keep the smallest body that triangulates well'))):
    x = 0.6 + nm * 3.2
    box(s, x, 5.0, 3.05, 1.75, fill=PANEL, name=f'Problem card {nm+1}')
    text(s, x + 0.2, 5.12, 2.65, 0.35, h, size=15, bold=True, color=CORAL, name=f'Problem {nm+1} heading')
    text(s, x + 0.2, 5.5, 2.65, 1.2, d, size=13, name=f'Problem {nm+1} text')
footer(s)
notes(s, 'There is no checkerboard, so the child is the calibration object: a bundle adjustment over the joints '
         'with bone-length constancy and left-right symmetry, which separates focal length from distance. Frames are '
         'chosen to spread the child across each image and across apparent sizes. After a first calibration the sync '
         'is refined geometrically and the cameras recalibrated (B010\'s focal lengths moved 1-2%). Reps the cameras '
         'cannot explain are set aside; one such rep had bent B199\'s whole calibration.')

# ================================================================ D6 triangulation
s = new_slide(); new.append(s)
title(s, 'Step 3: 3D keypoints, checked at every level')
# leave-one-out diagram
cx, cy = 3.2, 3.15
kid = box(s, cx - 0.45, cy - 0.45, 0.9, 0.9, fill=PANEL, shape=MSO_SHAPE.OVAL, name='LOO child')
shape_text(kid, '3D target', size=11, color=INK)
cams = [(cx, cy + 1.9, 'camera c\n(scored)', CORAL), (cx - 1.9, cy + 1.25, 'other\ncamera', TEAL),
        (cx + 1.9, cy + 1.25, 'other\ncamera', TEAL)]
for i, (px, py, lab, col) in enumerate(cams):
    line(s, cx, cy, px, py, color=col, width=2.5 if i else 1.5, dash=(i == 0), name=f'LOO line {i+1}')
for i, (px, py, lab, col) in enumerate(cams):
    b = box(s, px - 0.6, py - 0.3, 1.2, 0.6, fill=col, name=f'LOO camera {i+1}')
    shape_text(b, lab, size=11)
kid._element.getparent().remove(kid._element); s.shapes._spTree.append(kid._element)
text(s, 0.6, 1.55, 5.3, 0.9, [[('Leave-one-out target: ', {'bold': True}),
                                ('to score camera c, the 3D skeleton is triangulated from the other two cameras, so '
                                 'camera c\'s own detection errors can\'t flatter the result.', {})]],
     size=14, name='LOO text')
bl = [[('Robust triangulation: ', {'bold': True}),
       ('if one view disagrees (usually a left/right swap), use the best pair of views', {})],
      [('Scale: ', {'bold': True}), ('height measured in upright frames; each child scaled to their group\'s median', {})]]
text(s, 6.6, 1.55, 6.1, 1.7, bl, size=15, para_space=8, bullets=True, name='Triangulation bullets')
gates = [('Session', 'Calibration accuracy, camera roll, bone stability (≤8%), floor fit'),
         ('Joint', 'A bone >25% from the child\'s median loses both its joints'),
         ('Frame', 'At least 10 of the 12 limb joints kept'),
         ('Rep', 'Enough usable frames, no camera bumped, same child throughout')]
for i, (h, d) in enumerate(gates):
    x = 6.6 + (i % 2) * 3.1
    y = 3.45 + (i // 2) * 1.6
    box(s, x, y, 2.95, 1.45, fill=PANEL, name=f'Gate card {i+1}')
    text(s, x + 0.18, y + 0.12, 2.6, 0.35, h + ' gate', size=15, bold=True, color=CORAL, name=f'Gate {i+1} heading')
    text(s, x + 0.18, y + 0.5, 2.6, 0.9, d, size=12.5, name=f'Gate {i+1} text')
footer(s)
notes(s, 'Joints need confidence 0.5 and 15 px reprojection; a joint seen by two cameras gets no second chance. '
         'Stature: ear height above the floor plus 0.9 x ear width, in frames with trunk within 12 degrees, knees '
         'straight and ankles low. Scale: A 0.894, B 1.006, C 1.103 m (Korean growth charts). The all-camera '
         'skeleton is kept for proportions and stature but never used for scoring.')

# ================================================================ D7 outcome
s = new_slide(); new.append(s)
title(s, 'Outcome: 1,948 usable reps from 192 children')
cd = CategoryChartData()
cd.categories = ['Raw reps\n6,501', 'Synced\n5,080', 'Into the 3D build\n4,054', 'In usable sessions\n2,646',
                 'Usable reps\n1,948']
data = {'A': (2388, 1797, 860, 448, 345), 'B': (1551, 1456, 1442, 1131, 809), 'C': (2562, 1827, 1752, 1067, 794)}
for k, v in data.items():
    cd.add_series(f'Group {k}', v)
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_STACKED, Inches(0.6), Inches(1.45), Inches(7.6), Inches(5.0), cd)
gf.name = 'Funnel chart'
ch = gf.chart
style_chart(ch, 'Reps remaining at each stage')
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.TOP
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(12)
pl = ch.plots[0]
pl.gap_width = 55
pl.overlap = 100
for k, ser in zip(data, pl.series):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = SETC[k]
va = ch.value_axis
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = PANEL
va.format.line.fill.background()
va.tick_labels.font.size = Pt(10)
ch.category_axis.tick_labels.font.size = Pt(11)
ch.category_axis.tick_labels.font.color.rgb = INK
ch.category_axis.format.line.color.rgb = PALE
stat(s, 8.75, 1.5, 4.0, '9 px', 'median gap between a camera\'s 2D and its leave-one-out target (child ≈ 520 px tall)',
     name='Stat agreement')
stat(s, 8.75, 2.95, 4.0, '2 in 1 million', 'left/right swaps among scored joint pairs', name='Stat swaps')
text(s, 8.75, 4.4, 4.0, 0.35, 'Known gaps', size=16, bold=True, font=HEAD, name='Gaps heading')
gaps = [[('Bone-stability gate (8%) ', {'bold': True}),
         ('is among the reasons for 141 of 202 rejected sessions; it may be too strict, since OpenPose joints '
          'are not anatomical points', {})],
        [('A few moved-camera reps ', {'bold': True}), ('slip past the rep gate (fix identified)', {})]]
text(s, 8.75, 4.85, 4.0, 1.9, gaps, size=13, para_space=6, bullets=True, name='Gaps')
footer(s)
notes(s, 'Usable sessions: A 51/135, B 70/105, C 71/154. Group A drops sharply at "into the 3D build" because its '
         'stair tasks are excluded. Usable 3D frames: A 37k, B 130k, C 76k. The 9 px and swap figures are from the '
         'October audit of the set B build.')

# ---------------------------------------------------------------- reorder and renumber
lst = prs.slides._sldIdLst
ids = list(lst)
n_new = len(new)
moved = ids[-n_new:]
for el in moved:
    lst.remove(el)
for k, el in enumerate(moved):
    lst.insert(INSERT_AFTER + k, el)
for i, sl in enumerate(prs.slides, start=1):
    for sh in sl.shapes:
        if sh.name == 'Slide number' and sh.has_text_frame:
            r = sh.text_frame.paragraphs[0].runs[0]
            r.text = str(i)
prs.save(DECK)
print('saved', DECK, 'slides:', len(prs.slides))
