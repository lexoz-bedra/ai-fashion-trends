
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset

from ai_fashion_trends.vision.face import extract_face_or_center_crop
from ai_fashion_trends.vision.makeup_data import iter_labeled_makeup_paths
from ai_fashion_trends.vision.pair_classifier import SmokyBluePrediction

NUM_CLASSES = 3

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_torch_cache() -> None:
    if os.environ.get("TORCH_HOME"):
        return
    root = Path.cwd() / ".cache" / "torch"
    root.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(root)


def _resnet18_shell(num_classes: int = NUM_CLASSES) -> nn.Module:
    from torchvision.models import resnet18

    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_resnet18(num_classes: int = NUM_CLASSES) -> nn.Module:
    _ensure_torch_cache()
    from torchvision.models import ResNet18_Weights, resnet18

    w = ResNet18_Weights.IMAGENET1K_V1
    m = resnet18(weights=w)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def eval_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.12, 0.12, 0.08, 0.04),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class _MakeupCropDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, int]],
        transform: Any,
        *,
        use_mediapipe: bool,
    ) -> None:
        self.pairs = pairs
        self.transform = transform
        self.use_mediapipe = use_mediapipe

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, y = self.pairs[i]
        im = Image.open(path).convert("RGB")
        crop, _, _ = extract_face_or_center_crop(
            im, try_mediapipe=self.use_mediapipe
        )
        return self.transform(crop), y


def train_makeup_resnet(
    data_dir: Path,
    output_pt: Path,
    *,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    test_size: float = 0.2,
    seed: int = 42,
    use_mediapipe: bool = False,
    prob_threshold: float = 0.45,
) -> dict[str, Any]:
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    data_dir = data_dir.resolve()
    pairs = iter_labeled_makeup_paths(data_dir)

    if len(pairs) < 6:
        raise ValueError(f"Нужно больше изображений в {data_dir}, найдено: {len(pairs)}")

    ys = np.array([y for _, y in pairs], dtype=np.int64)
    if len(np.unique(ys)) < 2:
        raise ValueError("Нужны минимум два разных класса.")

    idx = np.arange(len(pairs))
    strat = ys if np.bincount(ys).min() >= 2 else None
    tr_i, va_i = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=strat
    )

    device = pick_device()
    torch.manual_seed(seed)

    train_ds = _MakeupCropDataset(
        pairs, train_transform(), use_mediapipe=use_mediapipe
    )
    full_eval = _MakeupCropDataset(
        pairs, eval_transform(), use_mediapipe=use_mediapipe
    )

    train_loader = DataLoader(
        Subset(train_ds, tr_i.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(full_eval, va_i.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_resnet18(NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.CrossEntropyLoss()

    best_state: dict[str, Any] | None = None
    best_acc = 0.0
    for _ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1).cpu().numpy()
                y_np = yb.numpy()
                correct += (pred == y_np).sum()
                total += len(y_np)
        acc = correct / max(total, 1)
        if acc >= best_acc:
            best_acc = float(acc)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    all_y: list[int] = []
    all_p: list[int] = []
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_np = yb.numpy()
            all_y.extend(y_np.tolist())
            all_p.extend(pred.tolist())

    output_pt = Path(output_pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state or model.state_dict(), output_pt)

    meta = {
        "kind": "resnet18_imagenet_finetune",
        "num_classes": NUM_CLASSES,
        "prob_threshold": prob_threshold,
        "label_order": ["smoky_eyes", "blue_eyeshadow", "other"],
        "epochs": epochs,
        "best_val_accuracy": best_acc,
        "data_dir": str(data_dir),
        "device_trained": str(device),
    }
    meta_path = output_pt.with_name(output_pt.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    report = classification_report(
        all_y, all_p, labels=[0, 1, 2], zero_division=0
    )

    return {
        "output_pt": str(output_pt),
        "meta_path": str(meta_path),
        "best_val_accuracy": best_acc,
        "classification_report": report,
        "n_train": len(tr_i),
        "n_val": len(va_i),
    }


class TorchMakeupClassifier:

    name = "resnet18_finetune"

    def __init__(
        self,
        model: nn.Module,
        *,
        meta: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model.eval()
        self.meta = meta
        self.device = device
        self.prob_threshold = float(meta.get("prob_threshold", 0.45))
        self._eval_tf = eval_transform()

    @classmethod
    def load(cls, pt_path: Path | str, device: torch.device | None = None) -> TorchMakeupClassifier:
        pt_path = Path(pt_path)
        meta_path = pt_path.with_name(pt_path.stem + ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dev = device or pick_device()
        model = _resnet18_shell(NUM_CLASSES)
        try:
            state = torch.load(pt_path, map_location=dev, weights_only=True)
        except TypeError:
            state = torch.load(pt_path, map_location=dev)
        model.load_state_dict(state)
        model.to(dev)
        return cls(model, meta=meta, device=dev)

    def predict_pil(
        self,
        image: Image.Image,
        *,
        use_mediapipe: bool = False,
    ) -> SmokyBluePrediction:
        crop, face_ok, _ = extract_face_or_center_crop(
            image.convert("RGB"), try_mediapipe=use_mediapipe
        )
        x = self._eval_tf(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
        p_smoky = float(proba[0])
        p_blue = float(proba[1])
        p_other = float(proba[2])
        conf = float(proba.max())
        if conf < self.prob_threshold:
            return SmokyBluePrediction(
                outcome="not_face",
                p_smoky=p_smoky,
                p_blue=p_blue,
                p_other=p_other,
                confidence=conf,
                face_detected=face_ok,
            )
        k = int(proba.argmax())
        out_map = {0: "smoky_eyes", 1: "blue_eyeshadow", 2: "other"}
        return SmokyBluePrediction(
            outcome=out_map[k],
            p_smoky=p_smoky,
            p_blue=p_blue,
            p_other=p_other,
            confidence=float(proba[k]),
            face_detected=face_ok,
        )
