import io
import time
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from dr_utils import CLASSES, load_any, predict_proba

EXAMPLES_DIR = Path("examples")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
N = len(CLASSES)

st.set_page_config(page_title="Diabetic Retinopathy Grading", page_icon="👁️", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 5rem; max-width: 1400px;}
.pred-title {text-align:center; font-size:2.2rem; font-weight:700; margin:.4rem 0 1.2rem;}
.bar-row {display:flex; justify-content:space-between; font-family:monospace; margin-top:.9rem;}
.bar-bg {height:5px; background:#374151; border-radius:3px; margin-top:4px;}
.bar-fg {height:5px; background:#F97316; border-radius:3px;}
div.stButton > button[kind="primary"] {background:#F97316; border:none; font-weight:600;}
div.stButton > button {border-radius:8px;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model…")
def get_model(path, device):
    return load_any(path, device)


def render_bars(probs):
    order = np.argsort(-probs)
    st.markdown(f'<div class="pred-title">{CLASSES[order[0]]}</div>', unsafe_allow_html=True)
    html = ""
    for i in order:
        p = probs[i] * 100
        html += (f'<div class="bar-row"><span>{i} · {CLASSES[i]}</span><span>{p:.0f}%</span></div>'
                 f'<div class="bar-bg"><div class="bar-fg" style="width:{p:.1f}%"></div></div>')
    st.markdown(html, unsafe_allow_html=True)


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


# ------------------------------ sidebar ------------------------------
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Device: **{device}**")

model, cfg = get_model(model_file, device)
st.caption(f"Loaded **{model_file}** · input {cfg['size']}×{cfg['size']} · "
           f"{'ImageNet-pretrained CNN + Ben Graham preprocessing' if cfg['kind']=='v2' else 'original custom CNN'}")
st.markdown("Used to grade **diabetic retinopathy** severity from fundus images")

tab_pred, tab_eval = st.tabs(["🔍 Classify", "📊 Evaluate"])

# ------------------------------ classify ------------------------------
with tab_pred:
    left, right = st.columns(2, gap="large")
    with left:
        up = st.file_uploader("img", type=[e[1:] for e in IMG_EXTS], key="single")
        examples = sorted(p for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in IMG_EXTS) if EXAMPLES_DIR.exists() else []
        if examples:
            st.caption("Examples")
            cols = st.columns(min(len(examples), 6))
            for i, p in enumerate(examples[:12]):
                with cols[i % len(cols)]:
                    st.image(str(p), use_container_width=True)
                    if st.button("Use", key=f"ex{i}"):
                        st.session_state["example"] = str(p)
        src = up if up is not None else st.session_state.get("example")
        img = Image.open(src).convert("RGB") if src else None
        if img is not None:
            st.image(img, use_container_width=True)
        run = st.button("Submit", type="primary", use_container_width=True, disabled=img is None)

    with right:
        with st.container(border=True):
            st.caption("Predictions")
            if run and img is not None:
                t0 = time.time()
                probs = predict_proba(model, cfg, [img], device, tta=tta)[0]
                dt = time.time() - t0
                render_bars(probs)
            else:
                st.info("Upload a fundus image and press **Submit**.")
        if run and img is not None:
            with st.container(border=True):
                st.text_input("Prediction time (s)", f"{dt:.5f}", disabled=True)
        st.caption("For education/research only — not a medical diagnosis.")

# ------------------------------ evaluate ------------------------------
with tab_eval:
    st.markdown(
        "Upload a **.zip** of labelled images, one folder per class, named by grade (`0`–`4`) "
        "or by class name, e.g. `0/abc.png`, `4/xyz.png` or `Moderate/def.png`."
    )
    zf = st.file_uploader("Test set (.zip)", type=["zip"], key="evalzip")
    if zf and st.button("Run evaluation", type="primary"):
        imgs, y = load_zip(zf)
        if len(imgs) == 0:
            st.error("No images found in folders named 0–4 or by class name.")
            st.stop()
        t0 = time.time()
        with st.spinner(f"Evaluating {len(imgs)} images…"):
            probs = predict_proba(model, cfg, imgs, device, tta=tta)
        el = time.time() - t0
        pred = probs.argmax(1)
        cm, prec, rec, f1, qwk = metrics(y, pred)
        present = cm.sum(1) > 0

        c = st.columns(5)
        c[0].metric("Accuracy", f"{(pred == y).mean()*100:.2f}%")
        c[1].metric("QWK", f"{qwk:.3f}")
        c[2].metric("Macro F1", f"{f1[present].mean()*100:.2f}%")
        c[3].metric("Images", len(imgs))
        c[4].metric("Avg time / image", f"{el/len(imgs)*1000:.0f} ms")

        a, b = st.columns(2, gap="large")
        with a:
            st.subheader("Confusion matrix")
            fig, ax = plt.subplots(figsize=(5.2, 4.6))
            fig.patch.set_facecolor("#0B1020"); ax.set_facecolor("#0B1020")
            ax.imshow(cm, cmap="Oranges")
            ax.set_xticks(range(N), CLASSES, rotation=40, ha="right", color="white")
            ax.set_yticks(range(N), CLASSES, color="white")
            ax.set_xlabel("Predicted", color="white"); ax.set_ylabel("True", color="white")
            for i in range(N):
                for j in range(N):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "#F97316")
            fig.tight_layout()
            st.pyplot(fig)
        with b:
            st.subheader("Per-class report")
            st.dataframe(pd.DataFrame({"Class": CLASSES, "Precision": prec.round(3), "Recall": rec.round(3),
                                       "F1": f1.round(3), "Support": cm.sum(1)}).set_index("Class"),
                         use_container_width=True)
            st.download_button(
                "Download predictions (CSV)",
                pd.DataFrame({"true": y, "pred": pred, "confidence": probs.max(1).round(4)}).to_csv(index=False),
                "predictions.csv", "text/csv")
