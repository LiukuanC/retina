"""
Diabetic retinopathy grading (Streamlit)

Design: "midnight clinical observatory" (Deep Iris canvas, cyan data accents,
Manrope standing in for Gilroy, 4px outlined boxes, pill buttons).

Flow
  1. Splash screen (line-art retina + ECG pulse) for ~3.5 s
  2. Splash dissolves and the app page rises into view
  3. Tabs: "Upload image" (classify + areas of interest) and "Upload zip" (evaluate)

Needs dr_utils.py (CLASSES, load_any, predict_proba) next to this file, and
.streamlit/config.toml (dark theme) for the best result.
"""
import base64
import io
import time
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps

from dr_utils import CLASSES, load_any, predict_proba

EXAMPLES_DIR = Path("examples")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
N = len(CLASSES)
GRID = 7                # occlusion grid (GRID x GRID patches) for the areas-of-interest map
SPLASH_SECONDS = 3.5    # how long the landing page stays before it dissolves

st.set_page_config(page_title="Diabetic Retinopathy Grading", page_icon="👁️", layout="wide")


# =====================================================================
# Small helpers
# =====================================================================
def html(s: str) -> str:
    """Strip indentation / blank lines so Markdown never turns HTML into a code block."""
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


def md(s: str):
    st.markdown(html(s), unsafe_allow_html=True)


def stretch(fn, *args, **kwargs):
    """Full-width widgets on both old (use_container_width) and new (width='stretch') Streamlit."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def to_b64(img: Image.Image, fmt="JPEG", quality=88) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": quality} if fmt == "JPEG" else {}))
    return base64.b64encode(buf.getvalue()).decode()


# =====================================================================
# Clinical copy (International Clinical DR severity scale, grades 0-4)
# =====================================================================
FINDINGS = {
    0: dict(
        summary="No visible signs of diabetic retinopathy.",
        signs=[("Clear retina", "No microaneurysms, haemorrhages or exudates.")],
        follow="Routine screening, usually every 12 months. Keep blood sugar, blood pressure and lipids in range.",
    ),
    1: dict(
        summary="Earliest stage, usually without symptoms.",
        signs=[("Microaneurysms", "Tiny red dots from weakened capillary walls. The only finding at this grade.")],
        follow="Repeat screening in about 6 to 12 months and review diabetes control.",
    ),
    2: dict(
        summary="More than just microaneurysms, but short of severe.",
        signs=[
            ("Microaneurysms", "Small red dots scattered across the retina."),
            ("Dot-and-blot haemorrhages", "Larger red spots from leaking vessels."),
            ("Hard exudates", "Yellow, sharp-edged lipid deposits, often in rings."),
            ("Cotton-wool spots", "Pale, fluffy patches from small areas of poor blood supply."),
        ],
        follow="Eye-care review within roughly 3 to 6 months. Macular swelling needs checking.",
    ),
    3: dict(
        summary="High risk of progressing to proliferative disease within a year.",
        signs=[
            ("Widespread haemorrhages", "Bleeding across all four quadrants of the retina."),
            ("Venous beading", "Veins that look like a string of beads."),
            ("IRMA", "Abnormal branching vessels inside the retina."),
        ],
        follow="Prompt referral to an ophthalmologist.",
    ),
    4: dict(
        summary="Advanced stage with new, fragile blood vessels.",
        signs=[
            ("Neovascularisation", "New abnormal vessels on the optic disc or elsewhere on the retina."),
            ("Pre-retinal or vitreous haemorrhage", "Blood in front of the retina or in the vitreous gel."),
            ("Fibrous tissue", "Scarring that can pull on and detach the retina."),
        ],
        follow="Urgent ophthalmology referral. There is a real risk of vision loss.",
    ),
}


# =====================================================================
# Line-art (original drawing, used on the splash and in the app header)
# =====================================================================
def retina_art(uid: str) -> str:
    return f"""
<svg class="art" viewBox="0 0 600 600" fill="none" stroke="#d6d9ff" stroke-width="2"
     stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
<defs>
<linearGradient id="sg-{uid}" x1="0" x2="1" y1="0" y2="0">
<stop offset="0" stop-color="#2ee9ff" stop-opacity="0"/>
<stop offset=".5" stop-color="#2ee9ff" stop-opacity=".9"/>
<stop offset="1" stop-color="#2ee9ff" stop-opacity="0"/>
</linearGradient>
<clipPath id="cp-{uid}"><circle cx="300" cy="300" r="248"/></clipPath>
</defs>
<circle class="draw" cx="300" cy="300" r="250"/>
<circle class="ring" cx="300" cy="300" r="214" stroke-dasharray="3 12" opacity=".7"/>
<circle class="draw d2" cx="300" cy="300" r="172" opacity=".45"/>
<g class="draw d1">
<path d="M410 290 C380 230 330 170 250 140"/>
<path d="M410 290 C370 262 310 240 200 245"/>
<path d="M410 290 C380 335 320 385 215 410"/>
<path d="M410 290 C395 360 360 440 300 490"/>
<path d="M330 195 C320 170 322 150 332 128"/>
<path d="M290 245 C270 230 255 215 250 195"/>
<path d="M300 375 C285 395 280 415 285 440"/>
<path d="M250 140 C230 140 215 145 195 160"/>
<circle cx="410" cy="290" r="34"/>
<circle cx="410" cy="290" r="16" opacity=".6"/>
<circle cx="230" cy="310" r="26" stroke-dasharray="2 7"/>
</g>
<g clip-path="url(#cp-{uid})"><rect class="scan" x="40" y="0" width="520" height="3" rx="1.5" fill="url(#sg-{uid})" stroke="none"/></g>
<g stroke="#2ee9ff" stroke-width="2.5">
<path class="draw d3" d="M20 80V20H80M520 20H580V80M580 520V580H520M80 580H20V520"/>
<rect class="roi-a" x="298" y="152" width="64" height="56" rx="7" stroke-dasharray="6 5"/>
<rect class="roi-b" x="226" y="352" width="64" height="56" rx="7" stroke-dasharray="6 5"/>
<rect class="roi-c" x="170" y="212" width="58" height="52" rx="7" stroke-dasharray="6 5"/>
</g>
<g fill="#2ee9ff" stroke="none" font-family="Manrope, sans-serif" font-weight="600" font-size="15">
<text class="roi-a" x="304" y="146">A</text>
<text class="roi-b" x="232" y="346">B</text>
<text class="roi-c" x="176" y="206">C</text>
</g>
</svg>
"""


# =====================================================================
# CSS: app
# =====================================================================
APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600&display=swap');
:root{
  --canvas:#383fea;                 /* sampled from your reference screenshot */
  --panel:rgba(255,255,255,.08);    /* card / field fill */
  --deep:#2a30c4;                   /* sidebar */
  --edge:rgba(255,255,255,.5);      /* button / pill outlines */
  --track:rgba(255,255,255,.22);    /* bar tracks, inactive strokes */
  --box:rgba(255,255,255,.4);       /* box outline colour */
  --line:4px;                       /* box outline thickness */
  --lilac:#d6d9ff; --cyan:#2ee9ff; --mint:#00ffaa; --white:#fff; --ash:#e4e6ff;
}
.stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp textarea,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp li, .stApp small,
.stApp [data-baseweb="select"] div, .stApp [data-baseweb="tab"] {
  font-family:'Manrope',ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif !important;
}
.stApp{background:var(--canvas) !important; color:var(--white); font-weight:500;}
header[data-testid="stHeader"]{background:transparent !important;}
[data-testid="stDecoration"], footer, [data-testid="stAppDeployButton"], .stDeployButton{display:none !important;}
.block-container{max-width:1200px; padding:1.6rem 1.5rem 4rem !important;}

/* sidebar: everything white */
[data-testid="stSidebar"]{background:var(--deep) !important; border-right:var(--line) solid var(--box);}
[data-testid="stSidebar"], [data-testid="stSidebar"] *{color:#fff !important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{letter-spacing:-.03em; font-weight:600;}

/* top bar */
.nav{display:flex; align-items:center; justify-content:flex-end; gap:16px; flex-wrap:wrap;}
.pills{display:flex; gap:8px; flex-wrap:wrap;}
.pill{display:inline-flex; align-items:center; gap:8px; border:3px solid var(--edge); border-radius:9999px;
      padding:.3rem .95rem; font-size:14px; letter-spacing:.02em; color:#fff; background:var(--panel);}
.pill .dot{width:8px; height:8px; border-radius:50%; background:var(--mint); box-shadow:0 0 8px var(--mint);}

/* hero */
.hero{display:grid; grid-template-columns:minmax(0,1fr) 210px; align-items:center; gap:32px; margin:22px 0 18px;}
.hero .art{width:210px; height:210px; display:block;}
.h-title{font-size:clamp(36px,5.4vw,58px); line-height:1; letter-spacing:-.045em; font-weight:600; color:#fff; margin:0 0 14px;}
.h-sub{font-size:17px; line-height:1.44; letter-spacing:-.01em; color:var(--ash); max-width:58ch; margin:0;}
.hl{display:inline-block; color:var(--cyan); border:3px dashed var(--cyan); border-radius:7px; padding:0 .14em .05em;}
@media (max-width:760px){.hero{grid-template-columns:1fr}.hero .art{display:none}}

/* tabs: white text, 6px top corners, 4px space around, no red underline */
.stTabs [data-baseweb="tab-list"]{gap:8px; padding:4px !important; border-bottom:var(--line) solid var(--box); background:transparent !important;}
.stTabs [data-baseweb="tab"]{height:auto !important; padding:4px !important; margin:0 !important;
  border-radius:6px 6px 0 0 !important; border:0 !important; box-shadow:none !important;
  background:transparent !important; font-size:16px; font-weight:500;}
.stTabs [data-baseweb="tab"][aria-selected="false"], .stTabs [data-baseweb="tab"][aria-selected="false"] *{color:#fff !important;}
.stTabs [data-baseweb="tab"][aria-selected="true"]{background:#fff !important;}
.stTabs [data-baseweb="tab"][aria-selected="true"], .stTabs [data-baseweb="tab"][aria-selected="true"] *{color:var(--canvas) !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{display:none !important; height:0 !important; background:transparent !important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.4rem;}

/* outlined cards */
[class*="st-key-card"]{
  border:var(--line) solid var(--box); border-radius:28px; padding:26px 26px 28px;
  background:var(--panel);
}
.card-title{font-size:24px; line-height:1.2; font-weight:600; letter-spacing:-.04em; color:#fff;}
.card-sub{font-size:14px; letter-spacing:.02em; color:var(--lilac); margin:2px 0 10px;}

/* inputs: white fields with #16165c text (works with or without config.toml) */
[data-baseweb="select"] > div, [data-testid="stSidebar"] [data-baseweb="select"] > div{
  background:#fff !important; border:3px solid var(--edge) !important; border-radius:16px !important;}
[data-baseweb="select"] *, [data-testid="stSidebar"] [data-baseweb="select"] *{color:#16165c !important; -webkit-text-fill-color:#16165c !important;}
[data-baseweb="select"] svg, [data-testid="stSidebar"] [data-baseweb="select"] svg{fill:#16165c !important; color:#16165c !important;}
[data-baseweb="popover"] ul, [data-baseweb="menu"]{background:#fff !important;}
[data-baseweb="popover"] li, [data-baseweb="popover"] li *, [data-baseweb="menu"] li, [data-baseweb="menu"] li *{color:#16165c !important;}
[data-baseweb="popover"] li:hover, [data-baseweb="menu"] li[aria-selected="true"]{background:#e4e6ff !important;}

[data-testid="stFileUploaderDropzone"]{
  background:#fff !important; border:var(--line) dashed var(--box) !important;
  border-radius:24px !important; padding:26px !important;}
[data-testid="stFileUploader"] *{color:#16165c !important;}
[data-testid="stFileUploader"] svg{fill:#16165c !important;}
[data-testid="stFileUploaderDropzone"] button{
  border-radius:9999px !important; background:#fff !important; border:3px solid #16165c !important;
  padding:.5rem 1.3rem !important; font-weight:600 !important;}
[data-testid="stFileUploaderDropzone"] button *{color:#16165c !important;}
[data-testid="stFileUploaderFile"]{background:#fff !important; border-radius:14px; padding:6px 10px;}
[data-testid="stImage"] img{border-radius:20px; border:3px solid var(--box);}

/* buttons */
.stButton > button, .stDownloadButton > button{
  border-radius:9999px; padding:.7rem 1.6rem; font-size:17px; font-weight:500; letter-spacing:-.01em;
  background:transparent; color:#fff; border:3px solid var(--edge); transition:border-color .2s, box-shadow .2s;}
.stButton > button:hover, .stDownloadButton > button:hover{border-color:var(--cyan); color:#fff;}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible{outline:3px solid var(--cyan); outline-offset:3px;}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"]{
  background:#fff; border-color:#fff; color:var(--canvas); box-shadow:0 0 22px rgba(255,255,255,.3);}
.stButton > button[kind="primary"] *, .stButton > button[data-testid="stBaseButton-primary"] *{color:var(--canvas) !important; font-weight:600;}
.stButton > button:disabled{opacity:.4; box-shadow:none;}

/* metrics (evaluation) */
[data-testid="stMetric"]{background:var(--panel); border:var(--line) solid var(--box); border-radius:24px; padding:16px 20px;}
[data-testid="stMetricLabel"] p{color:var(--ash); font-size:14px;}
[data-testid="stMetricValue"]{color:var(--cyan); font-weight:600; letter-spacing:-.04em;}

/* classification */
.grade-row{display:flex; align-items:stretch; justify-content:space-between; gap:16px; flex-wrap:wrap; margin:6px 0 18px;}
.grade-name{font-size:clamp(32px,4vw,46px); line-height:1; font-weight:600; letter-spacing:-.045em; color:#fff; margin:10px 0 0;}
.tag{display:inline-block; border:3px solid var(--edge); border-radius:9999px; padding:.2rem .85rem; font-size:14px; color:#fff;}
.tag.ok{border-color:var(--mint); color:var(--mint);}
.conf{min-width:150px; border:var(--line) solid var(--box); border-radius:24px; padding:12px 20px; background:rgba(255,255,255,.1);}
.conf b{display:block; font-size:clamp(38px,5vw,54px); line-height:1; font-weight:600; letter-spacing:-.05em; color:var(--cyan);}
.conf span{font-size:13px; letter-spacing:.02em; color:var(--ash);}
.meter{display:grid; grid-template-columns:repeat(var(--n),minmax(0,1fr)); gap:6px; margin:8px 0 18px;}
.seg{border:3px solid var(--track); border-radius:16px; padding:8px 6px 9px; text-align:center; color:#fff;}
.seg b{display:block; font-size:20px; font-weight:600; letter-spacing:-.03em;}
.seg span{display:block; font-size:11px; line-height:1.2; letter-spacing:.01em; margin-top:2px; overflow-wrap:anywhere;}
.seg.on{background:#fff; border-color:#fff; color:var(--canvas);}
.bar-row{display:flex; justify-content:space-between; gap:12px; font-size:15px; margin-top:14px; font-variant-numeric:tabular-nums;}
.bar-bg{height:8px; background:var(--track); border-radius:9999px; margin-top:6px; overflow:hidden;}
.bar-fg{height:100%; background:rgba(255,255,255,.7); border-radius:9999px;}
.bar-fg.top{background:var(--cyan);}
.stats{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; margin-top:20px;}
.stat{border:3px solid var(--track); border-radius:18px; padding:10px 12px;}
.stat b{display:block; font-size:18px; font-weight:600; letter-spacing:-.03em; color:#fff;}
.stat span{font-size:12px; letter-spacing:.02em; color:var(--lilac);}
.note{margin-top:16px; border:3px solid var(--track); border-radius:18px; padding:12px 16px; font-size:14px; line-height:1.44; color:var(--ash);}
.note.warn{border-color:var(--cyan); color:#fff;}
.empty{margin-top:6px; border:3px dashed var(--track); border-radius:20px; padding:18px 20px; color:var(--ash); line-height:1.44;}
.muted{color:var(--ash); font-size:15px; line-height:1.44;}
.divider{height:var(--line); border-radius:9999px; background:var(--track); margin:26px 0 20px;}

/* areas of interest (drawn on the uploaded image) */
.annot{position:relative; border-radius:22px; overflow:hidden; border:3px solid var(--box); line-height:0; background:#000; margin-top:12px;}
.annot img{width:100%; display:block;}
.annot img.heat{position:absolute; inset:0; height:100%;}
.roi{position:absolute; border:3px dashed var(--cyan); border-radius:7px; box-shadow:0 0 14px rgba(46,233,255,.55);}
.roi span{position:absolute; left:-3px; top:-27px; background:var(--cyan); color:#16165c; border-radius:9999px;
          min-width:24px; height:24px; line-height:24px; text-align:center; font-size:13px; font-weight:600;}
.lg-row{display:grid; grid-template-columns:30px minmax(0,1fr) 86px; align-items:center; gap:12px; margin-top:12px;}
.lg-tag{width:28px; height:28px; border-radius:50%; background:var(--cyan); color:#16165c; font-weight:600; text-align:center; line-height:28px; font-size:14px;}
.lg-row b{display:block; font-size:16px; font-weight:600; color:#fff;}
.lg-row small{font-size:13px; color:var(--lilac);}
.lg-bar{height:8px; background:var(--track); border-radius:9999px; overflow:hidden;}
.lg-bar i{display:block; height:100%; background:var(--cyan); border-radius:9999px;}
.find{display:grid; grid-template-columns:14px minmax(0,1fr); gap:12px; margin-top:14px;}
.find i{width:10px; height:10px; border-radius:3px; background:var(--cyan); margin-top:7px;}
.find b{display:block; font-size:17px; font-weight:600; letter-spacing:-.02em; color:#fff;}
.find small{font-size:14px; line-height:1.44; color:var(--ash);}
.follow{margin-top:20px; border:var(--line) solid var(--box); border-radius:22px; padding:14px 18px; background:rgba(255,255,255,.1);}
.follow b{font-size:15px; font-weight:600; color:#fff;}
.follow p{margin:4px 0 0; font-size:15px; line-height:1.44; color:var(--ash);}
.disclaimer{margin-top:28px; font-size:13px; letter-spacing:.02em; color:var(--lilac); text-align:center;}
/* --- fixes: selected model text and uploaded-file icon --- */
html body [data-testid="stSidebar"] [data-testid="stSelectbox"] *,
html body [data-testid="stSelectbox"] [data-baseweb="select"] *,
html body [data-testid="stSidebar"] [data-baseweb="select"] input{
  color:#16165c !important; -webkit-text-fill-color:#16165c !important; opacity:1 !important; caret-color:#16165c;}
html body [data-testid="stSelectbox"] [data-baseweb="select"] svg{fill:#16165c !important;}
html body [data-testid="stFileUploaderFileIcon"],
html body [data-testid="stFileUploaderFile"] > div:first-child:not([data-testid="stFileUploaderFileData"]),
html body [data-testid="stFileUploaderDropzoneInstructions"] :has(> svg){
  background:transparent !important; border:3px solid #16165c !important; border-radius:10px !important;}
html body [data-testid="stFileUploader"] svg{fill:#16165c !important; color:#16165c !important;}
html body [data-testid="stFileChip"] > div:first-child,
html body [data-testid="stFileUploader"] div:has(> svg:only-child):not(button div){
  background:transparent !important; background-color:transparent !important;
  border:3px solid #16165c !important; border-radius:10px !important; box-shadow:none !important;}
html body [data-testid="stFileChip"] svg{fill:#16165c !important; color:#16165c !important;}
</style>
"""


# =====================================================================
# CSS: splash screen (first run only)
# =====================================================================
SPLASH_CSS = """
<style>
.splash{position:fixed; inset:0; z-index:2147483000; display:flex; flex-direction:column; overflow:hidden;
  padding:clamp(18px,3vw,40px); color:#fff;
  background:#383fea;
  animation:splashOut 1.1s cubic-bezier(.7,0,.25,1) __T__ forwards;}
@keyframes splashOut{from{opacity:1; transform:translateY(0); visibility:visible}
                     to{opacity:0; transform:translateY(-9vh); visibility:hidden}}
.st-key-page{animation:pageRise 1.2s cubic-bezier(.2,.8,.2,1) __T__ both;}
@keyframes pageRise{from{opacity:0; transform:translateY(9vh)} to{opacity:1; transform:none}}

.splash .s-nav{display:flex; align-items:center; justify-content:flex-end; position:relative; z-index:2;}
.splash .pill{border:3px solid rgba(255,255,255,.5); border-radius:9999px; padding:.3rem 1rem; font-size:14px; color:#fff; letter-spacing:.02em; background:rgba(255,255,255,.08);}
.splash .stage{flex:1; display:grid; grid-template-columns:minmax(0,.9fr) minmax(0,1.1fr); align-items:center;
  gap:clamp(16px,4vw,72px); width:100%; max-width:1320px; margin:0 auto; position:relative; z-index:2;}
.splash .art-wrap{display:flex; justify-content:center;}
.splash .art{width:100%; max-width:min(620px,70vh); height:auto; display:block;}
.splash .title{font-size:clamp(44px,7.2vw,110px); line-height:.92; font-weight:600; letter-spacing:-.065em; margin:0 0 26px;
  animation:rise 1s cubic-bezier(.2,.8,.2,1) .25s both;}
.splash .hl{display:inline-block; color:#2ee9ff; border:3px dashed #2ee9ff; border-radius:7px; padding:0 .12em .06em;}
.splash .lede{font-size:clamp(16px,1.5vw,19px); line-height:1.44; letter-spacing:-.01em; color:#fff; max-width:46ch; margin:0;
  animation:rise 1s cubic-bezier(.2,.8,.2,1) .6s both;}
.splash .ecg{position:absolute; left:0; right:0; bottom:17%; width:100%; height:clamp(60px,12vh,120px); z-index:1; opacity:.8;
  filter:drop-shadow(0 0 8px rgba(46,233,255,.8));}
.splash .ecg path{stroke-dasharray:2400; animation:draw 2.6s cubic-bezier(.4,0,.2,1) .7s both;}
.splash .boot{position:relative; z-index:2; display:flex; align-items:center; gap:16px; font-size:14px; letter-spacing:.02em; color:#d6d9ff;}
.splash .boot i{position:relative; display:block; flex:1; height:4px; border-radius:9999px; background:rgba(255,255,255,.25); overflow:hidden;}
.splash .boot i::after{content:""; position:absolute; inset:0; background:#2ee9ff; transform-origin:left;
  animation:fill __T__ linear both;}

.splash .draw{stroke-dasharray:2000; animation:draw 2.4s cubic-bezier(.4,0,.2,1) both;}
.splash .d1{animation-delay:.3s} .splash .d2{animation-delay:.6s} .splash .d3{animation-delay:.9s}
.splash .ring{transform-box:fill-box; transform-origin:center; animation:spin 48s linear infinite;}
.splash .scan{animation:scan 2.6s ease-in-out 1.2s infinite alternate;}
.splash .roi-a{animation:pop .6s ease-out 2.0s both} .splash .roi-b{animation:pop .6s ease-out 2.3s both} .splash .roi-c{animation:pop .6s ease-out 2.6s both}

@keyframes draw{from{stroke-dashoffset:2000} to{stroke-dashoffset:0}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes scan{from{transform:translateY(60px)} to{transform:translateY(540px)}}
@keyframes pop{from{opacity:0} to{opacity:1}}
@keyframes fill{from{transform:scaleX(0)} to{transform:scaleX(1)}}
@keyframes rise{from{opacity:0; transform:translateY(24px)} to{opacity:1; transform:none}}
@keyframes splashFade{from{opacity:1; visibility:visible} to{opacity:0; visibility:hidden}}

@media (max-width:820px){
  .splash .stage{grid-template-columns:1fr; align-content:center;}
  .splash .art-wrap{position:absolute; right:-28%; top:4%; width:96%; opacity:.28; z-index:0;}
  .splash .copy{position:relative; z-index:2;}
}
@media (prefers-reduced-motion: reduce){
  .splash, .splash *, .st-key-page{animation:none !important;}
  .splash{animation:splashFade .6s linear 2s forwards !important;}
}
</style>
"""


def splash_html() -> str:
    css = SPLASH_CSS.replace("__T__", f"{SPLASH_SECONDS}s")
    return f"""{css}
<div class="splash" role="presentation">
<div class="s-nav"><span class="pill">Research preview</span></div>
<svg class="ecg" viewBox="0 0 1200 120" preserveAspectRatio="none" fill="none" stroke="#2ee9ff" stroke-width="2.5"
     stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
<path d="M0 70H420L440 70L458 28L488 108L514 8L540 70H610L626 56L642 70H1200"/>
</svg>
<div class="stage">
<div class="art-wrap">{retina_art("s")}</div>
<div class="copy">
<div class="title">Reading the retina in <span class="hl">seconds.</span></div>
<p class="lede">Diabetic retinopathy grading from a single retina photo. Get a severity grade from 0 to 4 and see the areas that shaped it.</p>
</div>
</div>
<div class="boot"><span>Starting up</span><i></i></div>
</div>
"""


# =====================================================================
# Model + analysis
# =====================================================================
@st.cache_resource(show_spinner="Loading model…")
def get_model(path, device):
    return load_any(path, device)


POS_ROW = ["Upper", "Central", "Lower"]
POS_COL = ["left", "centre", "right"]


def place_name(cy: float, cx: float) -> str:
    r = min(2, int(cy * 3))
    c = min(2, int(cx * 3))
    if r == 1 and c == 1:
        return "Centre"
    if r == 1:
        return f"Central {POS_COL[c]}"
    return f"{POS_ROW[r]} {POS_COL[c]}"


def analyse_regions(model, cfg, img, device, target):
    """Occlusion sensitivity: hide each patch in turn and see how much the predicted grade's
    probability drops. Works with any classifier; no access to model internals needed."""
    base = img.copy()
    base.thumbnail((512, 512))
    w, h = base.size
    fill = tuple(int(v) for v in np.asarray(base).reshape(-1, 3).mean(0))
    batch = [base]
    for gy in range(GRID):
        for gx in range(GRID):
            im = base.copy()
            im.paste(fill, (gx * w // GRID, gy * h // GRID, (gx + 1) * w // GRID, (gy + 1) * h // GRID))
            batch.append(im)
    p = predict_proba(model, cfg, batch, device, tta=False)
    sal = np.clip(p[0, target] - p[1:, target], 0, None).reshape(GRID, GRID)
    peak = float(sal.max())

    out = dict(base=to_b64(base), heat=None, regions=[])
    if peak < 0.03:          # no single area matters: the grade rests on the whole image
        return out

    # soft heat map
    a = np.asarray(Image.fromarray((sal / peak * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)) / 255.0
    cmap = LinearSegmentedColormap.from_list("heat", ["#5a62ff", "#2ee9ff", "#a8fbff", "#ffffff"])
    rgba = cmap(a)
    rgba[..., 3] = np.clip((a - 0.15) / 0.85, 0, 1) ** 1.1 * 0.8
    out["heat"] = to_b64(Image.fromarray((rgba * 255).astype(np.uint8), "RGBA"), "PNG")

    # up to three separate peaks
    work = sal.copy()
    for _ in range(3):
        y, x = np.unravel_index(work.argmax(), work.shape)
        if work[y, x] < 0.35 * peak:
            break
        cx, cy = (x + 0.5) / GRID, (y + 0.5) / GRID
        half = 0.75 / GRID
        x0, y0 = max(0.0, cx - half), max(0.0, cy - half)
        x1, y1 = min(1.0, cx + half), min(1.0, cy + half)
        out["regions"].append(dict(x=x0, y=y0, w=x1 - x0, h=y1 - y0, rel=float(work[y, x] / peak),
                                   name=place_name(cy, cx)))
        work[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = 0
    return out


# =====================================================================
# Render helpers
# =====================================================================
def severity_meter(active=None) -> str:
    segs = "".join(
        f'<div class="seg{" on" if i == active else ""}"><b>{i}</b><span>{c}</span></div>'
        for i, c in enumerate(CLASSES)
    )
    return f'<div class="meter" style="--n:{N}">{segs}</div>'


def prob_bars(probs) -> str:
    order = np.argsort(-probs)
    rows = ""
    for rank, i in enumerate(order):
        p = probs[i] * 100
        rows += (f'<div class="bar-row"><span>{i} · {CLASSES[i]}</span><span>{p:.0f}%</span></div>'
                 f'<div class="bar-bg"><div class="bar-fg{" top" if rank == 0 else ""}" style="width:{p:.1f}%"></div></div>')
    return rows


def result_html(res) -> str:
    probs = res["probs"]
    top = int(np.argmax(probs))
    conf = probs[top] * 100
    if top == 0:
        tag = '<span class="tag ok">No signs detected</span>'
    else:
        tag = f'<span class="tag">Grade {top} of {N - 1}</span>'
    note = ""
    if conf < 60:
        note = ('<div class="note warn">Low confidence. Upload a sharper, evenly lit retina photo '
                'before relying on this grade.</div>')
    w, h = res["size"]
    return f"""
<div class="grade-row">
<div>{tag}<div class="grade-name">{CLASSES[top]}</div></div>
<div class="conf"><b>{conf:.0f}%</b><span>confidence</span></div>
</div>
{severity_meter(top)}
{prob_bars(probs)}
<div class="stats">
<div class="stat"><b>{res["dt"]:.4f} s</b><span>Prediction time</span></div>
<div class="stat"><b>{w}×{h}</b><span>Image size</span></div>
<div class="stat"><b>{"On" if res["tta"] else "Off"}</b><span>Flip averaging</span></div>
</div>
{note}
"""


def annotated_html(res, show_heat, show_boxes) -> str:
    heat = (f'<img class="heat" alt="" src="data:image/png;base64,{res["heat"]}">'
            if show_heat and res.get("heat") else "")
    boxes = ""
    if show_boxes:
        for k, r in enumerate(res["regions"]):
            boxes += (f'<div class="roi" style="left:{r["x"] * 100:.1f}%;top:{r["y"] * 100:.1f}%;'
                      f'width:{r["w"] * 100:.1f}%;height:{r["h"] * 100:.1f}%"><span>{"ABC"[k]}</span></div>')
    return (f'<div class="annot"><img alt="Uploaded image with highlighted regions" '
            f'src="data:image/jpeg;base64,{res["base"]}">{heat}{boxes}</div>')


def legend_html(regions) -> str:
    rows = ""
    for k, r in enumerate(regions):
        level = "High" if r["rel"] >= 0.75 else "Medium" if r["rel"] >= 0.5 else "Lower"
        rows += (f'<div class="lg-row"><span class="lg-tag">{"ABC"[k]}</span>'
                 f'<div><b>{r["name"]}</b><small>{level} influence</small></div>'
                 f'<div class="lg-bar"><i style="width:{r["rel"] * 100:.0f}%"></i></div></div>')
    return rows


def findings_html(top) -> str:
    f = FINDINGS[min(top, 4)]
    items = "".join(f'<div class="find"><i></i><div><b>{n}</b><small>{d}</small></div></div>' for n, d in f["signs"])
    return (f'<div class="muted">{f["summary"]}</div>{items}'
            f'<div class="follow"><b>Typical follow-up</b><p>{f["follow"]}</p></div>')


# =====================================================================
# Evaluation helpers (unchanged logic)
# =====================================================================
def folder_to_label(name):
    n = name.strip().lower().replace("_", " ")
    if n.isdigit() and int(n) < N:
        return int(n)
    for i, c in enumerate(CLASSES):
        if n == c.lower():
            return i
    return None


def load_zip(file):
    imgs, labels = [], []
    with zipfile.ZipFile(file) as z:
        for name in z.namelist():
            p = Path(name)
            if p.suffix.lower() in IMG_EXTS and len(p.parts) >= 2 and "__MACOSX" not in name:
                lab = folder_to_label(p.parts[-2])
                if lab is not None:
                    imgs.append(Image.open(io.BytesIO(z.read(name))).convert("RGB"))
                    labels.append(lab)
    return imgs, np.array(labels)


def metrics(y, p):
    cm = np.zeros((N, N), dtype=int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(N), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(N), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(N), where=(prec + rec) > 0)
    w = np.array([[(i - j) ** 2 for j in range(N)] for i in range(N)]) / (N - 1) ** 2
    exp = np.outer(cm.sum(1), cm.sum(0)) / cm.sum()
    qwk = 1 - (w * cm).sum() / (w * exp).sum() if (w * exp).sum() > 0 else 0.0
    return cm, prec, rec, f1, qwk


# =====================================================================
# Page
# =====================================================================
md(APP_CSS)

# --- 1 + 2: splash on the first run of a session, then the page rises in
if not st.session_state.get("splashed"):
    md(splash_html())
    st.session_state["splashed"] = True

# --- sidebar: original settings, kept
with st.sidebar:
    st.header("⚙️ Settings")
    available = [p for p in ("best_v2b.pth", "best_v2.pth", "best.pth") if Path(p).exists()]
    if not available:
        st.error("Put best.pth and/or best_v2.pth next to app.py.")
        st.stop()
    model_file = st.selectbox("Model weights", available,
                              help="best_v2.pth = improved model (train_v2.py); best.pth = original CNN")
    tta = st.checkbox("Test-time augmentation (flip ×3)", value=True,
                      help="Slower, usually a little more accurate.")
    explain = st.checkbox("Highlight areas of interest", value=True,
                          help=f"Runs {GRID * GRID} extra passes per image to find which regions shaped the grade.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Device: **{device}**")

model, cfg = get_model(model_file, device)
with st.sidebar:
    st.caption(f"Loaded **{model_file}** · input {cfg['size']}×{cfg['size']} · "
               f"{'ImageNet-pretrained CNN + Ben Graham preprocessing' if cfg['kind'] == 'v2' else 'original custom CNN'}")

with st.container(key="page"):
    md(f"""
    <div class="nav">
      <div class="pills">
        <span class="pill"><i class="dot"></i>{model_file}</span>
        <span class="pill">{device.upper()}</span>
        <span class="pill">Research use only</span>
      </div>
    </div>
    <div class="hero">
      <div>
        <div class="h-title">Grade the retina in one <span class="hl">upload.</span></div>
        <p class="h-sub">Drop in a retina photograph to get a diabetic retinopathy severity grade from 0 to {N - 1},
        plus the regions of the image that shaped the result.</p>
      </div>
      {retina_art("h")}
    </div>
    """)

    tab_img, tab_zip = st.tabs(["Upload image", "Upload zip"])

    # ------------------------------------------------------------ upload image
    with tab_img:
        left, right = st.columns(2, gap="large")

        with left:
            with st.container(key="card_upload"):
                md('<div class="card-title">Upload image</div>'
                   '<div class="card-sub">JPG, PNG, TIFF, BMP or WebP</div>')
                up = st.file_uploader("Upload image", type=[e[1:] for e in IMG_EXTS], key="single",
                                      label_visibility="collapsed")

                examples = (sorted(p for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in IMG_EXTS)
                            if EXAMPLES_DIR.exists() else [])
                if examples:
                    st.caption("Or try an example")
                    cols = st.columns(4)
                    for i, p in enumerate(examples[:8]):
                        with cols[i % 4]:
                            stretch(st.image, str(p))
                            if st.button("Use", key=f"ex{i}"):
                                st.session_state["example"] = str(p)

                if up is not None:
                    src, sig = up, ("upload", up.name, up.size)
                elif st.session_state.get("example"):
                    src, sig = st.session_state["example"], ("example", st.session_state["example"])
                else:
                    src, sig = None, None
                img = ImageOps.exif_transpose(Image.open(src)).convert("RGB") if src else None
                if img is not None:
                    sig = sig + (model_file, tta, explain)

                run = stretch(st.button, "Classify image", type="primary", disabled=img is None, key="run")

                if run and img is not None:
                    with st.spinner("Analysing image…" + (" The area map takes a few seconds." if explain else "")):
                        t0 = time.time()
                        probs = predict_proba(model, cfg, [img], device, tta=tta)[0]
                        dt = time.time() - t0
                        res = dict(sig=sig, probs=probs, dt=dt, size=img.size, tta=tta, base=None, heat=None, regions=[])
                        if explain and int(np.argmax(probs)) > 0:
                            res.update(analyse_regions(model, cfg, img, device, int(np.argmax(probs))))
                    st.session_state["result"] = res

                res = st.session_state.get("result")
                if res is not None and res["sig"] != sig:
                    res = None          # new image or changed settings: ask for a fresh classification

                # The preview is the uploaded image itself. After classifying, the areas of
                # interest are drawn right on it, so there is no second image box.
                if img is not None:
                    top = int(np.argmax(res["probs"])) if res is not None else None
                    if res is not None and top > 0 and explain and res.get("base"):
                        t1, t2 = st.columns(2)
                        show_heat = t1.toggle("Heat map", value=True, key="t_heat")
                        show_boxes = t2.toggle("Region boxes", value=True, key="t_box")
                        md(annotated_html(res, show_heat, show_boxes))
                        if res["regions"]:
                            md(legend_html(res["regions"]))
                        else:
                            md('<div class="note">No single area dominates. The grade rests on the whole image.</div>')
                        md('<div class="note">Regions mark where hiding part of the photo changes the answer most. '
                           'They point to areas worth reviewing and are not confirmed lesions.</div>')
                    else:
                        stretch(st.image, img)
                        if res is not None and top == 0:
                            md('<div class="note">No areas of concern were flagged for this image.</div>')
                        elif res is not None and not explain:
                            md('<div class="note">Turn on <b>Highlight areas of interest</b> in the sidebar and '
                               'classify again to see them on the image.</div>')

        with right:
            with st.container(key="card_result"):
                md(f'<div class="card-title">Classification</div><div class="card-sub">Severity grade 0 to {N - 1}</div>')
                if res is not None:
                    md(result_html(res))
                    top = int(np.argmax(res["probs"]))
                    md(f'<div class="divider"></div><div class="card-title">What grade {top} usually shows</div>'
                       '<div class="card-sub">Typical findings, not detections</div>' + findings_html(top))
                else:
                    md(severity_meter(None) +
                       '<div class="empty">Upload an image, then choose <b>Classify image</b>. '
                       'The grade, confidence and class probabilities appear here.</div>')

        md('<div class="disclaimer">For education and research only. Not a medical diagnosis.</div>')

    # ------------------------------------------------------------ upload zip
    with tab_zip:
        with st.container(key="card_zip"):
            md('<div class="card-title">Evaluate a labelled test set</div>'
               '<div class="card-sub">One folder per class, named by grade (0 to 4) or class name</div>'
               '<p class="muted">For example <code>0/abc.png</code>, <code>4/xyz.png</code> or '
               '<code>Moderate/def.png</code> inside a single .zip file.</p>')
            zf = st.file_uploader("Test set (.zip)", type=["zip"], key="evalzip", label_visibility="collapsed")
            go = st.button("Run evaluation", type="primary", disabled=zf is None)

        if zf and go:
            imgs, y = load_zip(zf)
            if len(imgs) == 0:
                st.error("No images found in folders named 0 to 4 or by class name.")
            else:
                t0 = time.time()
                with st.spinner(f"Evaluating {len(imgs)} images…"):
                    probs = predict_proba(model, cfg, imgs, device, tta=tta)
                el = time.time() - t0
                pred = probs.argmax(1)
                cm, prec, rec, f1, qwk = metrics(y, pred)
                present = cm.sum(1) > 0

                c = st.columns(5)
                c[0].metric("Accuracy", f"{(pred == y).mean() * 100:.2f}%")
                c[1].metric("QWK", f"{qwk:.3f}")
                c[2].metric("Macro F1", f"{f1[present].mean() * 100:.2f}%")
                c[3].metric("Images", len(imgs))
                c[4].metric("Avg time / image", f"{el / len(imgs) * 1000:.0f} ms")

                m1, m2 = st.columns(2, gap="large")
                with m1:
                    with st.container(key="card_cm"):
                        md('<div class="card-title">Confusion matrix</div><div class="card-sub">Rows are true grade, columns are predicted</div>')
                        fig, ax = plt.subplots(figsize=(5.4, 4.8))
                        fig.patch.set_alpha(0)
                        ax.set_facecolor("none")
                        ax.imshow(cm, cmap=LinearSegmentedColormap.from_list("iris", ["#2a30c4", "#5a62ff", "#2ee9ff"]))
                        ax.set_xticks(range(N), CLASSES, rotation=40, ha="right", color="white")
                        ax.set_yticks(range(N), CLASSES, color="white")
                        ax.set_xlabel("Predicted", color="white")
                        ax.set_ylabel("True", color="white")
                        for s in ax.spines.values():
                            s.set_color("#4846c6")
                        for i in range(N):
                            for j in range(N):
                                ax.text(j, i, cm[i, j], ha="center", va="center", color="white")
                        fig.tight_layout()
                        st.pyplot(fig, transparent=True)
                        plt.close(fig)
                with m2:
                    with st.container(key="card_report"):
                        md('<div class="card-title">Per-class report</div><div class="card-sub">Precision, recall and F1 for each grade</div>')
                        df = pd.DataFrame({"Class": CLASSES, "Precision": prec.round(3), "Recall": rec.round(3),
                                           "F1": f1.round(3), "Support": cm.sum(1)}).set_index("Class")
                        stretch(st.dataframe, df)
                        st.download_button(
                            "Download predictions (CSV)",
                            pd.DataFrame({"true": y, "pred": pred, "confidence": probs.max(1).round(4)}).to_csv(index=False),
                            "predictions.csv", "text/csv")
