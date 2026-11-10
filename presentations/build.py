"""Build the supervisor progress deck (python-pptx; every element is a native, movable object)."""
from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION

HERE = Path(__file__).parent
IMG = HERE / "img"
OUT = HERE / "child_lifting_progress.pptx"

INK = RGBColor(0x1F, 0x2A, 0x36)
MUTED = RGBColor(0x5B, 0x6B, 0x7A)
PANEL = RGBColor(0xEE, 0xF1, 0xF4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xC9, 0xD3, 0xDD)
CORAL = RGBColor(0xE4, 0x57, 0x2E)
CFG = {  # match the mesh colours in the render videos
    "C0": RGBColor(0xE0, 0x7A, 0x6F),
    "C8": RGBColor(0xD9, 0xA9, 0x2E),
    "C14": RGBColor(0x5D, 0xA5, 0xD6),
    "C19": RGBColor(0x5F, 0xB0, 0x5F),
}
HEAD, BODY = "Cambria", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W = 13.333


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = color


def text(slide, x, y, w, h, runs, size=16, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None, para_space=0):
    """runs: str, or list of paragraphs; each paragraph a str or list of (text, {opts})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if para_space and i:
            p.space_before = Pt(para_space)
        segs = para if isinstance(para, list) else [(para, {})]
        for seg, opts in segs:
            r = p.add_run()
            r.text = seg
            r.font.name = opts.get("font", font)
            r.font.size = Pt(opts.get("size", size))
            r.font.bold = opts.get("bold", bold)
            r.font.italic = opts.get("italic", False)
            r.font.color.rgb = opts.get("color", color)
    return tb


def title(slide, t, color=INK):
    text(slide, 0.6, 0.45, W - 1.2, 0.9, t, size=32, bold=True, font=HEAD, color=color,
         anchor=MSO_ANCHOR.MIDDLE, name="Title")


def footer(slide, n, dark=False):
    c = PALE if dark else MUTED
    text(slide, 0.6, 7.0, 6, 0.3, "Child 3D lifting · progress update", size=10, color=c, name="Footer")
    text(slide, W - 1.6, 7.0, 1.0, 0.3, str(n), size=10, color=c, align=PP_ALIGN.RIGHT, name="Slide number")


def box(slide, x, y, w, h, fill=PANEL, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=None, radius=0.06):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if name:
        s.name = name
    return s


def badge(slide, x, y, label, fill=CORAL, d=0.5, size=16, name=None):
    s = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL, name=name)
    tf = s.text_frame
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = p.add_run()
    r.text = label
    r.font.size, r.font.bold, r.font.name = Pt(size), True, BODY
    r.font.color.rgb = WHITE
    return s


def picture(slide, path, x, y, w=None, h=None, name=None, alt=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    p = slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)
    if name:
        p.name = name
    if alt:
        p._element.nvPicPr.cNvPr.set("descr", alt)
    return p


def stat(slide, x, y, w, big, label, color=CORAL, name=None, size=36):
    text(slide, x, y, w, 0.8, big, size=size, bold=True, font=HEAD, color=color, name=name)
    text(slide, x, y + 0.8, w, 0.7, label, size=14, color=MUTED, name=(name + " label") if name else None)


def style_chart(chart):
    chart.has_legend = False
    chart.font.name = BODY
    chart.font.size = Pt(12)
    chart.font.color.rgb = MUTED


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


# ---- images derived from the renders ----
Image.open(IMG / "B010_GMS_1_1_cam1_C0_C8_C14_C19.jpg").crop((940, 604, 1866, 1150)).save(IMG / "hero_c19.jpg", quality=92)

# ===================== 1. Title =====================
s = prs.slides.add_slide(BLANK)
bg(s, INK)
text(s, 0.8, 2.0, 6.6, 2.2, "Lifting children's movement into 3D from a single camera",
     size=40, bold=True, font=HEAD, color=WHITE, name="Title")
text(s, 0.8, 4.35, 6.4, 1.0, "Progress update: child body shape, joint definitions and head orientation",
     size=20, color=PALE, name="Subtitle")
text(s, 0.8, 5.6, 6, 0.4, "November 2026", size=14, color=PALE, name="Date")
picture(s, IMG / "hero_c19.jpg", 7.6, 1.55, w=5.2, name="Hero render",
        alt="Child-shaped mesh placed against the triangulated skeleton, camera and side views")
notes(s, "Ten-minute update on the child 3D lifting work. The question throughout: how close can we get to "
         "three-camera ground truth using one camera and an adult-trained pose model?")

# ===================== 2. Problem =====================
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Adult body models can't produce a child's proportions")
picture(s, IMG / "shape_tpose.png", 0.6, 1.55, w=8.7, name="T-pose comparison",
        alt="MotionBERT adult body, fitted adult body and fitted kid body for child B010")
text(s, 0.6, 5.35, 8.7, 0.9,
     "Child B010, aged 3–4. Grey skeleton = the child's measured bone lengths. Left: MotionBERT's adult body "
     "shrunk to the child's height. Middle: adult model forced to fit. Right: adult model plus a child template.",
     size=12, color=MUTED, name="Caption")
stat(s, 9.6, 1.7, 3.3, "1.76 vs 1.41", "leg-to-torso ratio: adult body vs this child", name="Stat ratio")
stat(s, 9.6, 3.3, 3.3, "33 → 7 mm", "bone-length error: adult body vs child-template fit", name="Stat bones")
text(s, 9.6, 4.9, 3.1, 1.3, "Forcing the adult model to fit pushed its shape settings far outside the normal "
     "range and produced distorted bodies.", size=14, color=INK, name="Stat note")
footer(s, 2)
notes(s, "MotionBERT is trained on adults, so its body has adult proportions: legs too long for a 4-year-old. "
         "Fitting the adult SMPL model to the child's bones only worked by driving the shape parameters to about "
         "plus or minus 6, giving distorted heads and shoulders and the knee 5 cm too low. Adding the AGORA kid "
         "template as one extra shape direction fixes this with ordinary parameter values.")

# ===================== 3. Pipeline and scoring =====================
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "How the pipeline works, and how we score it")
steps = [
    ("2D keypoints", "OpenPose finds the joints in each video frame"),
    ("Lift to 3D", "MotionBERT estimates the 3D pose (trained on adults)"),
    ("Child body", "Body model fitted to the child's bone lengths"),
    ("Place in room", "Body moved so it lines up with the image and floor"),
    ("Refine", "Joint angles adjusted to the 2D, with foot and balance rules"),
]
cw, gap, x0, y0 = 2.25, 0.21, 0.6, 1.75
for i, (h, d) in enumerate(steps):
    x = x0 + i * (cw + gap)
    box(s, x, y0, cw, 2.0, name=f"Step {i+1} card")
    badge(s, x + 0.2, y0 + 0.2, str(i + 1), name=f"Step {i+1} badge")
    text(s, x + 0.2, y0 + 0.82, cw - 0.4, 0.4, h, size=17, bold=True, name=f"Step {i+1} heading")
    text(s, x + 0.2, y0 + 1.22, cw - 0.4, 0.75, d, size=13, color=MUTED, name=f"Step {i+1} text")
    if i < len(steps) - 1:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + cw + 0.02), Inches(y0 + 0.9),
                               Inches(gap - 0.04), Inches(0.22))
        a.fill.solid(); a.fill.fore_color.rgb = PALE; a.line.fill.background(); a.name = f"Arrow {i+1}"
text(s, 0.6, 4.2, 6, 0.4, "How we measure success", size=20, bold=True, font=HEAD, name="Scoring heading")
cards = [
    ("Ground truth", "Three cameras triangulated, leaving out the camera being tested"),
    ("Main metric", "Placement error: mm between each predicted joint and the truth, no alignment"),
    ("Fair testing", "Tuned on 10 dev children (30 trials); 29 test children kept untouched"),
]
cw2 = (W - 1.2 - 2 * 0.3) / 3
for i, (h, d) in enumerate(cards):
    x = 0.6 + i * (cw2 + 0.3)
    box(s, x, 4.75, cw2, 1.45, fill=WHITE, name=f"Score card {i+1}").line.color.rgb = PALE
    text(s, x + 0.25, 4.95, cw2 - 0.5, 0.4, h, size=16, bold=True, color=CORAL, name=f"Score {i+1} heading")
    text(s, x + 0.25, 5.38, cw2 - 0.5, 1.1, d, size=14, name=f"Score {i+1} text")
footer(s, 3)
notes(s, "Steps 3 to 5 are where this work sits. Ground truth comes from the three Korea cameras, triangulated "
         "with the tested camera left out so it can't mark its own homework. The split is by child: 10 dev "
         "children for decisions, 29 test children reported once at the end. Adults from BioCV are only used "
         "for training and calibration.")

# ===================== 4. Progression chart =====================
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Each change brought the 3D body closer to the truth")
cd = CategoryChartData()
cd.categories = ["Original", "Child body shape", "Feet & balance", "Calibrated hips & shoulders"]
cd.add_series("Placement error (mm)", (93.9, 62.5, 56.0, 48.9))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.5), Inches(8.0), Inches(4.9), cd)
gf.name = "Progression chart"
ch = gf.chart
style_chart(ch)
ch.has_title = True
ch.chart_title.text_frame.text = "Placement error, median of 10 dev children (mm)"
tp = ch.chart_title.text_frame.paragraphs[0]
tp.runs[0].font.size, tp.runs[0].font.bold, tp.runs[0].font.color.rgb = Pt(14), False, INK
plot = ch.plots[0]
plot.gap_width = 60
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format, dl.number_format_is_linked = "0", False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size, dl.font.bold, dl.font.color.rgb = Pt(16), True, INK
for pt_i, key in enumerate(["C0", "C8", "C14", "C19"]):
    pt = plot.series[0].points[pt_i]
    pt.format.fill.solid(); pt.format.fill.fore_color.rgb = CFG[key]
va = ch.value_axis
va.maximum_scale, va.minimum_scale = 100, 0
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = PANEL
va.format.line.fill.background()
va.visible = False
ch.category_axis.format.line.color.rgb = PALE
ch.category_axis.tick_labels.font.size = Pt(13)
ch.category_axis.tick_labels.font.color.rgb = INK
text(s, 0.6, 6.45, 8.0, 0.4, "Each bar includes all the changes to its left.", size=12, color=MUTED,
     name="Chart note")
stat(s, 9.3, 1.6, 3.4, "−48%", "placement error, 94 → 49 mm", name="Stat reduction")
stat(s, 9.3, 3.25, 3.4, "10 / 10", "dev children better than with the child body shape alone", name="Stat wins")
stat(s, 9.3, 4.9, 3.4, "43 → 27 mm", "pose error once the skeletons are aligned", name="Stat PA")
footer(s, 4)
notes(s, "Dev set, 10 children, 30 trials, median over children. Original = MotionBERT body placed by PnP "
         "(93.9 mm). Child body shape = kid template plus 2D refinement, config C8 (62.5). Feet & balance = C14: "
         "feet held at the floor height measured on the mesh, centre of mass kept over the feet, SMPL-24 joints "
         "and a girth prior on the shape (56.0). Calibrated hips & shoulders = C19, a learned mapping from the "
         "mesh to where OpenPose puts those joints (48.9). PA error (shape of the pose only) 42.6 to 26.9 mm.")

# ===================== 5. Video =====================
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "What the changes look like on one trial")
vw = 8.1
vh = vw * 1150 / 1866
mv = s.shapes.add_movie(str(IMG / "b010.mp4"), Inches(0.6), Inches(1.5), Inches(vw), Inches(vh),
                        poster_frame_image=str(IMG / "B010_GMS_1_1_cam1_C0_C8_C14_C19.jpg"), mime_type="video/mp4")
mv.name = "Comparison video"
text(s, 0.6, 1.5 + vh + 0.1, vw, 0.4, "Child B010. Each pair: camera view, then side view. White = ground truth.",
     size=12, color=MUTED, name="Video caption")
legend = [("C0", "Original", "197 mm"), ("C8", "Child body shape", "66 mm"),
          ("C14", "Feet & balance", "45 mm"), ("C19", "Calibrated hips & shoulders", "43 mm")]
text(s, 9.6, 1.55, 3.2, 0.4, "Error on this trial", size=16, bold=True, font=HEAD, name="Legend heading")
for i, (k, lab, v) in enumerate(legend):
    y = 2.15 + i * 0.95
    box(s, 9.6, y + 0.06, 0.32, 0.32, fill=CFG[k], shape=MSO_SHAPE.OVAL, name=f"Legend dot {k}")
    text(s, 10.1, y, 2.7, 0.4, lab, size=15, name=f"Legend {k}")
    text(s, 10.1, y + 0.36, 2.7, 0.4, v, size=15, bold=True, color=CFG[k] if k != "C8" else INK,
         name=f"Legend {k} value")
text(s, 9.6, 6.0, 3.2, 0.8, "The adult body drifts in depth; the child versions stay on the skeleton.",
     size=13, color=MUTED, name="Legend note")
footer(s, 5)
notes(s, "Click to play. The original adult body (red) is far too tall and drifts towards and away from the "
         "camera, which the camera view hides but the side view shows. The child body (yellow) fixes the "
         "proportions. Firm feet and balance (blue) stop the depth drift. Calibrated joints (green) tighten the "
         "hips and shoulders.")

# ===================== 6. Joint definitions =====================
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "OpenPose's “hip” is not the body model's hip joint")
pw = 7.6
picture(s, IMG / "A184_GMS_4_2_fit_az90.jpg", 0.6, 1.55, w=pw, name="Hip definition render",
        alt="Toddler mesh with OpenPose hips (white), model hip joints (yellow) and learned positions (green)")
text(s, 0.6, 1.55 + pw * 720 / 1280 + 0.1, pw, 0.4,
     "White = OpenPose · Yellow = model hip joint · Green = learned position (toddler, set A)",
     size=12, color=MUTED, name="Image caption")
pts = [
    [("The mismatch. ", {"bold": True}),
     ("OpenPose marks the hip on the body's outline. In young children it sits about 50 mm above the model's hip joint.", {})],
    [("The fix. ", {"bold": True}),
     ("We learned where OpenPose's hips and shoulders sit on the mesh, from 180 children and 11 adults. "
      "The mapping adjusts with the child's body proportions.", {})],
    [("The result. ", {"bold": True}),
     ("Hip error 25 → 17 mm on unseen children; whole-body placement error 56 → 49 mm.", {})],
]
text(s, 8.7, 1.65, 4.0, 4.8, pts, size=15, para_space=14, name="Explanation")
footer(s, 6)
notes(s, "The body model's joints and OpenPose's keypoints share names but not positions. We first tried "
         "correcting with OpenPose-minus-mocap offsets measured on adults; that failed because the model's "
         "joints aren't where mocap's are either. Instead we fitted the model to triangulated OpenPose in 180 "
         "Korea children and 11 BioCV adults and learned a vertex regressor whose weights change with the "
         "child-template weight. Cross-validated hip error for set B fell from 24.6 to 16.8 mm. Set A toddlers "
         "were dropped from training since they are outside the age range we care about.")

# ===================== 7. Head =====================
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "The head tips forward unless the face is in the fit")
cd = CategoryChartData()
cd.categories = ["No face points", "Face, ¼ weight", "Face, calibrated points", "Face, full weight"]
cd.add_series("Head pitch error (deg)", (-12.0, -0.2, 2.2, 3.2))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(1.5), Inches(7.6), Inches(4.6), cd)
gf.name = "Head pitch chart"
ch = gf.chart
style_chart(ch)
ch.has_title = True
ch.chart_title.text_frame.text = "Head pitch error vs truth (degrees; negative = tipped down)"
tp = ch.chart_title.text_frame.paragraphs[0]
tp.runs[0].font.size, tp.runs[0].font.bold, tp.runs[0].font.color.rgb = Pt(14), False, INK
plot = ch.plots[0]
plot.gap_width = 70
plot.series[0].invert_if_negative = False
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format, dl.number_format_is_linked = "+0.0;−0.0;0.0", False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size, dl.font.bold, dl.font.color.rgb = Pt(15), True, INK
for i, c in enumerate([CORAL, CFG["C19"], PALE, PALE]):
    pt = plot.series[0].points[i]
    pt.format.fill.solid(); pt.format.fill.fore_color.rgb = c
from pptx.oxml.ns import qn
from lxml import etree
for dpt in plot.series[0]._element.findall(qn("c:dPt")):
    if dpt.find(qn("c:invertIfNegative")) is None:
        el = etree.SubElement(dpt, qn("c:invertIfNegative"))
        el.set("val", "0")
        dpt.remove(el)
        dpt.find(qn("c:idx")).addnext(el)
ca = ch.category_axis
ca.reverse_order = True
ca.tick_labels.font.size, ca.tick_labels.font.color.rgb = Pt(13), INK
ca.format.line.color.rgb = PALE
from pptx.enum.chart import XL_TICK_LABEL_POSITION
ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
va = ch.value_axis
va.minimum_scale, va.maximum_scale = -15, 6
va.has_major_gridlines = False
va.visible = False
stat(s, 9.0, 1.6, 3.7, "~12° down", "head tilt in the current best body recipe", name="Stat tilt")
stat(s, 9.0, 3.2, 3.7, "49 vs 50 mm", "body error without vs with the face points", name="Stat cost")
text(s, 9.0, 4.85, 3.7, 1.6, "A light face term levels the head for about 1 mm of body accuracy.",
     size=15, name="Head takeaway")
footer(s, 7)
notes(s, "Dev set. Without face points (C19) the refinement bends the upper spine to hit the shoulders and the "
         "head follows, ending about 12 degrees down. Adding nose and ears to the fit levels it: quarter weight "
         "(C24) -0.2 degrees, calibrated face points at full weight (C25) +2.2, full weight with standard mesh "
         "points (C21) +3.2. Body placement error 48.9 for C19 vs 50.0 for C24 and 50.1 for C25. Scoring uses "
         "face points calibrated to OpenPose; the standard ones made correct heads look 4-7 degrees up.")

# ===================== 8. Next steps =====================
s = prs.slides.add_slide(BLANK)
bg(s, INK)
title(s, "Decision and next steps", color=WHITE)
items = [
    ("Decide", "Best body (C19) or level head for ~1 mm (C24)?"),
    ("Retrain MotionBERT", "Fine-tune on Korea children: about 155k training frames already triangulated"),
    ("Switch to YOLO", "The likely product detector; test the transfer on BioCV adults, which have video"),
    ("Report the test set", "Run the 29 untouched children once the recipe is fixed"),
]
cw, ch_, gx, gy = 5.9, 1.9, 0.33, 0.33
for i, (h, d) in enumerate(items):
    x = 0.6 + (i % 2) * (cw + gx)
    y = 1.6 + (i // 2) * (ch_ + gy)
    box(s, x, y, cw, ch_, fill=RGBColor(0x2C, 0x3A, 0x4A), name=f"Next card {i+1}")
    badge(s, x + 0.3, y + 0.3, str(i + 1), name=f"Next badge {i+1}")
    text(s, x + 1.0, y + 0.33, cw - 1.3, 0.45, h, size=20, bold=True, font=HEAD, color=WHITE, name=f"Next {i+1} heading")
    text(s, x + 1.0, y + 0.9, cw - 1.3, 1.1, d, size=16, color=PALE, name=f"Next {i+1} text")
footer(s, 8, dark=True)
notes(s, "The decision I'd like input on is the first card. The fine-tuning plan uses the triangulated Korea "
         "children with leave-one-camera-out targets, pose-only supervision and the original training data mixed "
         "in; OpenPose input first, YOLO only if that works. Korea has no RGB video, so YOLO has to be tested on "
         "BioCV.")

# ===================== 9. Appendix table =====================
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
title(s, "Appendix: all configurations on the dev set")
rows = [
    ("Config", "What changes", "Placed (mm)", "PA (mm)", "Better than C8"),
    ("step4", "Original: MotionBERT body placed by PnP", "93.9", "42.6", "0 / 10"),
    ("C8", "Child body shape, refined against the 2D", "62.5", "32.4", "–"),
    ("C14", "SMPL-24 joints, girth prior, firm feet, balance", "56.0", "31.1", "7 / 10"),
    ("C19", "C14 + calibrated OpenPose hips and shoulders", "48.9", "26.9", "10 / 10"),
    ("C20", "C19 + hip and shoulder widths in the shape fit", "48.7", "26.7", "10 / 10"),
    ("C21", "C19 + nose and ears in the fit", "50.3", "27.0", "10 / 10"),
    ("C23", "C21, face at ½ weight", "50.3", "27.0", "10 / 10"),
    ("C24", "C21, face at ¼ weight", "50.0", "26.9", "10 / 10"),
    ("C25", "C21 with calibrated face points", "50.1", "26.9", "10 / 10"),
]
tbl = s.shapes.add_table(len(rows), 5, Inches(0.6), Inches(1.5), Inches(W - 1.2), Inches(0.42 * len(rows))).table
s.shapes[-1].name = "Config table"
widths = [1.2, 6.5, 1.6, 1.4, 1.43]
for j, wd in enumerate(widths):
    tbl.columns[j].width = Inches(wd)
for i, r in enumerate(rows):
    for j, v in enumerate(r):
        c = tbl.cell(i, j)
        c.fill.solid()
        c.fill.fore_color.rgb = INK if i == 0 else (PANEL if i % 2 == 0 else WHITE)
        c.margin_left = c.margin_right = Inches(0.1)
        tf = c.text_frame
        tf.text = v
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if j < 2 else PP_ALIGN.RIGHT
        f = p.runs[0].font
        f.name, f.size = BODY, Pt(14)
        f.bold = i == 0 or r[0] == "C19"
        f.color.rgb = WHITE if i == 0 else INK
text(s, 0.6, 5.85, W - 1.2, 0.5, "Median across 10 dev children (30 trials). PA = error after aligning the "
     "skeletons, so it measures pose shape only.", size=12, color=MUTED, name="Table note")
footer(s, 9)
notes(s, "Backup only. C20's widths add 0.2 mm for an extra term, so C19 was kept. C22 (no girth prior) is "
         "not shown: it scored the same on joints but let the waist and chest drift by up to 170 mm.")

prs.save(OUT)
print("saved", OUT)
