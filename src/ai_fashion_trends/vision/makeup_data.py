
from __future__ import annotations

from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

SMOKY_DIRS = ("smoky_eyes",)
BLUE_DIRS = ("blue_eyeshadow", "blue-eyeshadow")
OTHER_DIRS = ("other",)


LABEL_NAMES = ("smoky_eyes", "blue_eyeshadow", "other")


def iter_labeled_makeup_paths(root: Path) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    root = Path(root)
    for p in root.iterdir():
        if not p.is_dir():
            continue
        name = p.name.lower()
        if name in {d.lower() for d in SMOKY_DIRS}:
            y = 0
        elif name in {d.lower() for d in BLUE_DIRS}:
            y = 1
        elif name in {d.lower() for d in OTHER_DIRS}:
            y = 2
        else:
            continue
        for f in sorted(p.rglob("*")):
            if f.suffix.lower() in IMAGE_EXTS:
                rows.append((f, y))
    return rows
