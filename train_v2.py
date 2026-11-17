"""Improved retinopathy training (Windows-safe). Produces best_v2.pth (loadable by app.py)."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from dr_utils import CLASSES, NORM, ben_preprocess, build_model

# ---------------- config ----------------
ARCH = "resnet50"      # or "efficientnet_b0"
IMG_SIZE = 512         # try 512 if GPU memory allows
BS, EPOCHS, LR, PATIENCE, SEED = 16, 30, 1e-4, 8, 5566
NUM_WORKERS = 2        # set to 0 if you still have multiprocessing problems
CLASS_WEIGHTS = True   # up-weight rare grades (3, 4) in the loss
WD = 1e-2              # weight decay (was 1e-4); more regularisation vs overfitting
AREA_RESIZE = True    # anti-aliased resize: robust to input resolution
CKPT = "best_v2b.pth"  # output file, so best_v2.pth is not overwritten
ROOT = Path("kaggle_retina")
MAKE_SUBMISSION = True
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"


class DRDataset(Dataset):
    """Must live at module level so Windows worker processes can import it."""

    def __init__(self, imgs, labels=None, aug=False):
        self.imgs, self.labels = imgs, labels
        if aug:
            self.tf = T.Compose([
                T.ToPILImage(), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                T.RandomRotation(180, fill=128),
                T.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
                T.ColorJitter(0.2, 0.2), T.ToTensor(), NORM])
        else:
            self.tf = T.Compose([T.ToTensor(), NORM])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = self.tf(self.imgs[i])
        return x if self.labels is None else (x, int(self.labels[i]))


def load_cache(ids, folder):
    return [ben_preprocess(Image.open(ROOT / folder / f"{i}.png"), IMG_SIZE, AREA_RESIZE)
            for i in tqdm(ids, desc=folder)]


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # ---------------- data ----------------
    df = pd.read_csv(ROOT / "train.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED,
                                        stratify=df["diagnosis"])
    tr_imgs = load_cache(train_df["id_code"], "train")
    va_imgs = load_cache(val_df["id_code"], "train")

    kw = dict(num_workers=NUM_WORKERS, pin_memory=device == "cuda",
              persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(DRDataset(tr_imgs, train_df["diagnosis"].values, aug=True),
                              BS, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(DRDataset(va_imgs, val_df["diagnosis"].values), BS, **kw)

    # ---------------- model / optim ----------------
    model = build_model(ARCH, 5, pretrained=True).to(device)
    w = None
    if CLASS_WEIGHTS:
        counts = np.bincount(train_df["diagnosis"], minlength=5)
        w = (counts.sum() / counts) ** 0.5          # sqrt inverse frequency (gentle)
        w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
        print("Class weights:", w.cpu().numpy().round(2))
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def run_epoch(loader, train):
        model.train(train)
        tot_loss, preds, ys = 0.0, [], []
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            with torch.set_grad_enabled(train), torch.autocast(device_type=device, enabled=use_amp):
                out = model(x)
                loss = loss_fn(out, y)
            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            tot_loss += loss.item()
            preds.append(out.argmax(1).cpu()); ys.append(y.cpu())
        p, y = torch.cat(preds).numpy(), torch.cat(ys).numpy()
        return tot_loss / len(loader), (p == y).mean(), cohen_kappa_score(y, p, weights="quadratic")

    # ---------------- train ----------------
    best_acc, bad = 0.0, 0
    for ep in range(EPOCHS):
        tl, ta, tk = run_epoch(train_loader, True)
        vl, va, vk = run_epoch(val_loader, False)
        scheduler.step()
        print(f"EP {ep:02d} train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f} QWK {vk:.3f}")
        if va > best_acc:
            best_acc, bad = va, 0
            torch.save({"arch": ARCH, "img_size": IMG_SIZE, "classes": CLASSES, "area": AREA_RESIZE,
                        "state_dict": model.state_dict()}, CKPT)
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stop"); break

    # ---------------- final eval with TTA ----------------
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["state_dict"])
    model.eval()

    @torch.no_grad()
    def predict_tta(loader):
        out = []
        for batch in loader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            p = sum(torch.softmax(model(v), 1) for v in (x, x.flip(-1), x.flip(-2))) / 3
            out.append(p.argmax(1).cpu())
        return torch.cat(out).numpy()

    vp = predict_tta(val_loader)
    print(f"\nVal acc (TTA): {(vp == val_df['diagnosis'].values).mean():.4f}  "
          f"QWK: {cohen_kappa_score(val_df['diagnosis'], vp, weights='quadratic'):.4f}")
    print(confusion_matrix(val_df["diagnosis"], vp))

    # ---------------- Kaggle submission ----------------
    if MAKE_SUBMISSION:
        sub = pd.read_csv(ROOT / "submission.csv")
        te_imgs = load_cache(sub["id_code"], "test")
        te_loader = DataLoader(DRDataset(te_imgs), BS, **kw)
        sub["diagnosis"] = predict_tta(te_loader)
        sub.to_csv("submission1.csv", index=False)
        print("Saved submission1.csv")


if __name__ == "__main__":
    main()
