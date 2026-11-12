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
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ----------------------------------------------------------------------------
# Model definition (must match training code exactly)
# ----------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding="same"),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding="same"),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.conv(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(28 * 28 * 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 5),
        )

    def forward(self, x):
        return self.classifier(self.flatten(self.encoder(x)))


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL_PATH = Path("best.pth")
EXAMPLES_DIR = Path("examples")
IMG_SIZE = 224  # 224 / 2^3 = 28 -> matches Linear(28*28*128, ...)
DEFAULT_CLASSES = "0, 1, 2, 3, 4"  # EDIT to your training label order
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

st.set_page_config(page_title="Ocular Disease Prediction", page_icon="👁️", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
.pred-title {text-align:center; font-size:2.2rem; font-weight:700; margin:.4rem 0 1.2rem;}
.bar-row {display:flex; justify-content:space-between; font-family:monospace; margin-top:.9rem;}
.bar-bg {height:5px; background:#374151; border-radius:3px; margin-top:4px;}
.bar-fg {height:5px; background:#F97316; border-radius:3px;}
.card {background:#1F2937; border:1px solid #374151; border-radius:10px; padding:1rem 1.2rem;}
div.stButton > button[kind="primary"] {background:#F97316; border:none; font-weight:600;}
div.stButton > button {border-radius:8px;}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str, device: str):
    model = CNN()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, nn.Module):  # whole model was pickled
        model = ckpt
    else:
        if isinstance(ckpt, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    ckpt = ckpt[key]
                    break
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
    return model.to(device).eval()


def build_transform(imagenet_norm: bool):
    t = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    if imagenet_norm:
        t.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(t)


@torch.no_grad()
def predict(model, imgs, tfm, device, batch=32):
    out = []
    for i in range(0, len(imgs), batch):
        x = torch.stack([tfm(im) for im in imgs[i : i + batch]]).to(device)
        out.append(torch.softmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(out)


def render_bars(classes, probs):
    order = np.argsort(-probs)
    st.markdown(f'<div class="pred-title">{classes[order[0]]}</div>', unsafe_allow_html=True)
    html = ""
    for i in order:
        p = probs[i] * 100
        html += (
            f'<div class="bar-row"><span>{classes[i]}</span><span>{p:.0f}%</span></div>'
            f'<div class="bar-bg"><div class="bar-fg" style="width:{p:.1f}%"></div></div>'
        )
    st.markdown(html, unsafe_allow_html=True)


def load_zip_dataset(file):
    """Zip layout: <class_name>/<image>.jpg  (folder name = true label)."""
    imgs, labels = [], []
    with zipfile.ZipFile(file) as z:
        for name in z.namelist():
            p = Path(name)
            if p.suffix.lower() in IMG_EXTS and len(p.parts) >= 2 and "__MACOSX" not in name:
                imgs.append(Image.open(io.BytesIO(z.read(name))).convert("RGB"))
                labels.append(p.parts[-2])
    return imgs, labels


def metrics(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(n), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(n), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n), where=(prec + rec) > 0)
    return cm, prec, rec, f1


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    class_str = st.text_area(
        "Class names (comma-separated, in training label order)",
        DEFAULT_CLASSES,
        help="Model has 5 outputs. Order must match the label indices used in training.",
    )
    classes = [c.strip() for c in class_str.split(",") if c.strip()]
    imagenet_norm = st.checkbox(
        "Use ImageNet normalization",
        value=False,
        help="Tick only if your training pipeline used Normalize(mean, std).",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Device: **{device}** · Input size: **{IMG_SIZE}×{IMG_SIZE}**")

if len(classes) != 5:
    st.error(f"The model outputs 5 classes, but {len(classes)} names were given.")
    st.stop()
if not MODEL_PATH.exists():
    st.error(f"`{MODEL_PATH}` not found. Put best.pth next to app.py.")
    st.stop()

model = load_model(str(MODEL_PATH), device)
tfm = build_transform(imagenet_norm)

st.markdown("Used to Predict Ocular diseases")
tab_pred, tab_eval = st.tabs(["🔍 Classify", "📊 Evaluate"])

# ----------------------------------------------------------------------------
# Tab 1 – classify
# ----------------------------------------------------------------------------
with tab_pred:
    left, right = st.columns(2, gap="large")

    with left:
        up = st.file_uploader("img", type=[e[1:] for e in IMG_EXTS], key="single")

        examples = sorted(p for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in IMG_EXTS) if EXAMPLES_DIR.exists() else []
        chosen = None
        if examples:
            st.caption("Examples")
            cols = st.columns(min(len(examples), 6))
            for i, p in enumerate(examples[:12]):
                with cols[i % len(cols)]:
                    st.image(str(p), use_container_width=True)
                    if st.button("Use", key=f"ex{i}"):
                        st.session_state["example"] = str(p)
            chosen = st.session_state.get("example")

        src = up if up is not None else chosen
        img = Image.open(src).convert("RGB") if src else None
        if img is not None:
            st.image(img, use_container_width=True)
        run = st.button("Submit", type="primary", use_container_width=True, disabled=img is None)

    with right:
        if run and img is not None:
            t0 = time.time()
            probs = predict(model, [img], tfm, device)[0]
            dt = time.time() - t0
            with st.container(border=True):
                st.caption("Predictions")
                render_bars(classes, probs)
            st.markdown('<div class="card" style="margin-top:1rem">', unsafe_allow_html=True)
            st.text_input("Prediction time (s)", f"{dt:.5f}", disabled=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            with st.container(border=True):
                st.caption("Predictions")
                st.info("Upload a fundus image and press **Submit**.")

# ----------------------------------------------------------------------------
# Tab 2 – evaluate
# ----------------------------------------------------------------------------
with tab_eval:
    st.markdown(
        "Upload a **.zip** of labelled test images, organised as one folder per class, "
        "e.g. `Cataract/img1.jpg`, `Normal/img2.jpg`. Folder names must match the class names in the sidebar."
    )
    zf = st.file_uploader("Test set (.zip)", type=["zip"], key="evalzip")

    if zf and st.button("Run evaluation", type="primary"):
        imgs, labels = load_zip_dataset(zf)
        lower = {c.lower(): i for i, c in enumerate(classes)}
        keep = [i for i, l in enumerate(labels) if l.lower() in lower]
        skipped = len(labels) - len(keep)
        if not keep:
            st.error("No folder names matched the class names. Check the sidebar class list.")
            st.stop()
        if skipped:
            st.warning(f"Skipped {skipped} images whose folder name didn't match any class.")

        imgs = [imgs[i] for i in keep]
        y_true = np.array([lower[labels[i].lower()] for i in keep])

        t0 = time.time()
        with st.spinner(f"Evaluating {len(imgs)} images…"):
            probs = predict(model, imgs, tfm, device)
        elapsed = time.time() - t0
        y_pred = probs.argmax(1)

        cm, prec, rec, f1 = metrics(y_true, y_pred, len(classes))
        acc = (y_true == y_pred).mean()
        present = cm.sum(1) > 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc*100:.2f}%")
        c2.metric("Macro F1", f"{f1[present].mean()*100:.2f}%")
        c3.metric("Images", len(imgs))
        c4.metric("Avg time / image", f"{elapsed/len(imgs)*1000:.1f} ms")

        a, b = st.columns(2, gap="large")
        with a:
            st.subheader("Confusion matrix")
            fig, ax = plt.subplots(figsize=(5, 4.5))
            fig.patch.set_facecolor("#0B1020")
            ax.set_facecolor("#0B1020")
            ax.imshow(cm, cmap="Oranges")
            ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right", color="white")
            ax.set_yticks(range(len(classes)), classes, color="white")
            ax.set_xlabel("Predicted", color="white")
            ax.set_ylabel("True", color="white")
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "#F97316")
            fig.tight_layout()
            st.pyplot(fig)
        with b:
            st.subheader("Per-class report")
            df = pd.DataFrame({
                "Class": classes,
                "Precision": prec.round(3),
                "Recall": rec.round(3),
                "F1": f1.round(3),
                "Support": cm.sum(1),
            }).set_index("Class")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download predictions (CSV)",
                pd.DataFrame({
                    "true": [classes[i] for i in y_true],
                    "pred": [classes[i] for i in y_pred],
                    "confidence": probs.max(1).round(4),
                }).to_csv(index=False),
                "predictions.csv",
                "text/csv",
            )
