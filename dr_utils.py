"""Shared code for training (train_v2.py) and the Streamlit app (app.py)."""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image

# APTOS / Kaggle diabetic-retinopathy grading, label index = severity
CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# ---------------- original model (best.pth) ----------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding="same"), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding="same"), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.conv(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(ConvBlock(3, 32), ConvBlock(32, 64), ConvBlock(64, 128))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(28 * 28 * 128, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 5),
        )

    def forward(self, x):
        return self.classifier(self.flatten(self.encoder(x)))


# ---------------- improved model (best_v2.pth) ----------------
def build_model(arch="resnet50", num_classes=5, pretrained=True):
    if arch == "resnet50":
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
    elif arch == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, num_classes))
    else:
        raise ValueError(arch)
    return m


# ---------------- preprocessing ----------------
def crop_black(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(gray > tol)
    if len(ys) == 0:
        return img
    return img[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def ben_preprocess(pil, size, area=False):
    """Crop black border -> resize -> subtract local average colour (Ben Graham).
    Makes lesions (microaneurysms, haemorrhages, exudates) much easier to see."""
    img = np.array(pil.convert("RGB"))
    img = crop_black(img)
    # INTER_AREA anti-aliases, so the result barely depends on the upload resolution
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA if area else cv2.INTER_LINEAR)
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), size / 30), -4, 128)


# ---------------- loading / inference (used by the app) ----------------
def load_any(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "arch" in ck:  # v2 checkpoint
        model = build_model(ck["arch"], pretrained=False)
        model.load_state_dict(ck["state_dict"])
        cfg = {"kind": "v2", "size": ck["img_size"], "area": ck.get("area", False)}
    else:  # original CNN state_dict
        if isinstance(ck, dict):
            for k in ("model_state_dict", "state_dict"):
                if k in ck:
                    ck = ck[k]
        model = CNN()
        model.load_state_dict({k.replace("module.", "", 1): v for k, v in ck.items()})
        cfg = {"kind": "v1", "size": 224}
    return model.to(device).eval(), cfg


def make_tensor(pil, cfg):
    if cfg["kind"] == "v2":
        return T.Compose([T.ToTensor(), NORM])(ben_preprocess(pil, cfg["size"], cfg.get("area", False)))
    return T.Compose([T.Resize((224, 224)), T.ToTensor()])(pil.convert("RGB"))


@torch.no_grad()
def predict_proba(model, cfg, imgs, device="cpu", tta=False, batch=16):
    out = []
    for i in range(0, len(imgs), batch):
        x = torch.stack([make_tensor(im, cfg) for im in imgs[i: i + batch]]).to(device)
        p = torch.softmax(model(x), 1)
        if tta:
            p = (p + torch.softmax(model(x.flip(-1)), 1) + torch.softmax(model(x.flip(-2)), 1)) / 3
        out.append(p.cpu().numpy())
    return np.concatenate(out)
