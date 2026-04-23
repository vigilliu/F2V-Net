#!/usr/bin/env python3
"""
Minimal two-point image labeler (no OpenCV required).

For each input image, click exactly two points in order:
1) center
2) keypoint

The script saves labels to each image's folder.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import List, Optional, Tuple

import tkinter as tk


Point = Tuple[float, float]
POINT_NAMES = ["center", "keypoint"]


def load_photoimage_with_fallback(image_path: Path) -> Tuple[tk.PhotoImage, Optional[str], str]:
    """
    Load image for Tk canvas.
    1) Try direct tk.PhotoImage
    2) If unsupported PNG encoding, convert by macOS sips then reload
    Returns: (photo, temp_file_path_if_any, source_desc)
    """
    try:
        photo = tk.PhotoImage(file=str(image_path))
        return photo, None, f"direct:{image_path.name}"
    except tk.TclError as first_err:
        # Some Tk builds cannot decode PNG (or this PNG encoding).
        # Try converting to Tk-friendly formats with macOS sips.
        attempts = [("bmp", ".bmp"), ("jpeg", ".jpg"), ("gif", ".gif"), ("ppm", ".ppm"), ("png", ".png")]
        errors: List[str] = [f"direct load failed: {first_err}"]
        for fmt, suffix in attempts:
            fd, tmp_name = tempfile.mkstemp(prefix=f"{image_path.stem}_tk_", suffix=suffix)
            os.close(fd)
            tmp_path = Path(tmp_name)
            cmd = ["sips", "-s", "format", fmt, str(image_path), "--out", str(tmp_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not tmp_path.exists():
                stderr = (result.stderr or "").strip() or (result.stdout or "").strip()
                errors.append(f"sips->{fmt} failed: {stderr}")
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                continue

            try:
                photo = tk.PhotoImage(file=str(tmp_path))
                return photo, str(tmp_path), f"sips:{fmt}"
            except tk.TclError as e:
                errors.append(f"tk load converted {fmt} failed: {e}")
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                continue

        raise RuntimeError(
            "Cannot open image in Tk (direct and converted formats all failed).\n"
            f"Image: {image_path}\n"
            + "\n".join(errors)
        ) from None


class TwoPointLabeler:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path
        self.points: List[Point] = []
        self.temp_image_path: Optional[str] = None
        self.image_source_desc: str = ""

        # Silence macOS Tk deprecation warning in terminal output.
        os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")
        self.root = tk.Tk()
        self.root.title(f"Label image: {self.image_path.name}")

        self.photo, self.temp_image_path, self.image_source_desc = load_photoimage_with_fallback(self.image_path)
        self.canvas = tk.Canvas(
            self.root,
            width=self.photo.width(),
            height=self.photo.height(),
            highlightthickness=0,
        )
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.info = tk.Label(
            self.root,
            text="Click point 1/2: center",
            anchor="w",
            justify="left",
            padx=8,
            pady=6,
            font=("Helvetica", 12),
        )
        self.info.pack(fill="x")
        print(
            f"Loaded image: {self.image_path.name} | source={self.image_source_desc} "
            f"| size={self.photo.width()}x{self.photo.height()}"
        )

        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<BackSpace>", self.undo_last_point)
        self.root.bind("<Escape>", self.cancel)
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)

        self.cancelled = False
        self.done = False

    def on_click(self, event: tk.Event) -> None:
        if self.done:
            return
        x, y = float(event.x), float(event.y)
        self.points.append((x, y))
        idx = len(self.points) - 1
        label = POINT_NAMES[idx]

        r = 4
        color = "red" if label == "center" else "yellow"
        self.canvas.create_oval(x - r, y - r, x + r, y + r, outline=color, fill=color, width=2)
        self.canvas.create_text(x + 10, y - 10, text=label, fill=color, anchor=tk.W, font=("Helvetica", 11, "bold"))

        if len(self.points) < 2:
            self.info.config(text=f"Click point 2/2: {POINT_NAMES[1]}")
            return

        self.done = True
        self.info.config(text="Done. Closing this window...")
        self.root.after(350, self.root.destroy)

    def undo_last_point(self, _event: tk.Event) -> None:
        if not self.points or self.done:
            return
        self.points.pop()
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        for i, (x, y) in enumerate(self.points):
            label = POINT_NAMES[i]
            r = 4
            color = "red" if label == "center" else "yellow"
            self.canvas.create_oval(x - r, y - r, x + r, y + r, outline=color, fill=color, width=2)
            self.canvas.create_text(x + 10, y - 10, text=label, fill=color, anchor=tk.W, font=("Helvetica", 11, "bold"))
        next_idx = len(self.points)
        self.info.config(text=f"Click point {next_idx + 1}/2: {POINT_NAMES[next_idx]}")

    def cancel(self) -> None:
        self.cancelled = True
        if self.temp_image_path:
            try:
                Path(self.temp_image_path).unlink(missing_ok=True)
            except OSError:
                pass
        self.root.destroy()

    def run(self) -> List[Point]:
        self.root.mainloop()
        if self.temp_image_path:
            try:
                Path(self.temp_image_path).unlink(missing_ok=True)
            except OSError:
                pass
        if self.cancelled:
            raise RuntimeError(f"Labeling cancelled: {self.image_path}")
        if len(self.points) != 2:
            raise RuntimeError(f"Expected 2 points, got {len(self.points)} for {self.image_path}")
        return self.points


def save_single_image_labels(image_path: Path, points: List[Point]) -> Path:
    labels = {
        "image": str(image_path),
        "point_order": POINT_NAMES,
        "points": [
            {"name": POINT_NAMES[i], "x": float(points[i][0]), "y": float(points[i][1])}
            for i in range(2)
        ],
    }
    out_path = image_path.parent / f"{image_path.stem}_labels.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    return out_path


def save_pair_labels(image_a: Path, points_a: List[Point], image_b: Path, points_b: List[Point]) -> Path:
    common_dir = image_a.parent if image_a.parent == image_b.parent else image_a.parent
    payload = {
        "point_order": POINT_NAMES,
        "images": [
            {
                "image": str(image_a),
                "points": [
                    {"name": "center", "x": float(points_a[0][0]), "y": float(points_a[0][1])},
                    {"name": "keypoint", "x": float(points_a[1][0]), "y": float(points_a[1][1])},
                ],
            },
            {
                "image": str(image_b),
                "points": [
                    {"name": "center", "x": float(points_b[0][0]), "y": float(points_b[0][1])},
                    {"name": "keypoint", "x": float(points_b[1][0]), "y": float(points_b[1][1])},
                ],
            },
        ],
    }
    out_path = common_dir / "pair_labels.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def parse_case_ids(case_ids: str) -> Optional[set[str]]:
    text = case_ids.strip()
    if not text:
        return None
    return {x.strip() for x in text.split(",") if x.strip()}


def collect_batch_cases(
    root_dir: Path,
    pred_name: str,
    tee_name: str,
    case_ids: Optional[set[str]],
) -> List[Tuple[str, Path, Path]]:
    cases: List[Tuple[str, Path, Path]] = []
    for p in sorted(root_dir.iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        case_name = p.name
        if case_ids is not None and case_name not in case_ids:
            continue
        pred_img = p / pred_name
        tee_img = p / tee_name
        if pred_img.exists() and tee_img.exists():
            cases.append((case_name, pred_img, tee_img))
    return cases


def should_skip_case(case_dir: Path, pred_name: str, tee_name: str, overwrite: bool) -> bool:
    if overwrite:
        return False
    pred_json = case_dir / f"{Path(pred_name).stem}_labels.json"
    tee_json = case_dir / f"{Path(tee_name).stem}_labels.json"
    pair_json = case_dir / "pair_labels.json"
    return pred_json.exists() and tee_json.exists() and pair_json.exists()


def run_single_case(case_name: str, image1: Path, image2: Path) -> None:
    print(f"\n=== Case {case_name} ===")
    print(f"Image 1: {image1}")
    print(f"Image 2: {image2}")
    print("Click order for each image: center -> keypoint")
    print("Hotkeys: Backspace=undo, Esc=cancel")

    labeler1 = TwoPointLabeler(image1)
    points1 = labeler1.run()
    out1 = save_single_image_labels(image1, points1)
    print(f"Saved: {out1}")

    labeler2 = TwoPointLabeler(image2)
    points2 = labeler2.run()
    out2 = save_single_image_labels(image2, points2)
    print(f"Saved: {out2}")

    pair_out = save_pair_labels(image1, points1, image2, points2)
    print(f"Saved: {pair_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label two points (center + keypoint) on two images.")
    parser.add_argument(
        "--image1",
        type=str,
        default="/Users/liujiacheng/Downloads/3dtee_3dct_DS4/49/pred_slice.png",
        help="First image path (used in single mode)",
    )
    parser.add_argument(
        "--image2",
        type=str,
        default="/Users/liujiacheng/Downloads/3dtee_3dct_DS4/49/tee_random_mid_256.png",
        help="Second image path (used in single mode)",
    )
    parser.add_argument(
        "--batch-root",
        type=str,
        default=None,
        help="Batch mode root directory, e.g. /Users/liujiacheng/Downloads/3dtee_3dct_DS4",
    )
    parser.add_argument(
        "--case-ids",
        type=str,
        default="",
        help="Optional comma-separated case folders, e.g. 49,50,51",
    )
    parser.add_argument(
        "--pred-name",
        type=str,
        default="pred_slice.png",
        help="Pred image filename in each case folder",
    )
    parser.add_argument(
        "--tee-name",
        type=str,
        default="tee_random_mid_256.png",
        help="TEE image filename in each case folder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_labels.json and pair_labels.json in batch mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_root:
        batch_root = Path(args.batch_root).expanduser().resolve()
        if not batch_root.exists():
            raise FileNotFoundError(f"Batch root not found: {batch_root}")
        if not batch_root.is_dir():
            raise NotADirectoryError(f"Batch root is not a directory: {batch_root}")

        case_ids = parse_case_ids(args.case_ids)
        cases = collect_batch_cases(
            root_dir=batch_root,
            pred_name=args.pred_name,
            tee_name=args.tee_name,
            case_ids=case_ids,
        )
        if not cases:
            raise RuntimeError("No valid case folders found (both pred and tee images are required).")

        print(f"Batch root: {batch_root}")
        print(f"Found cases: {len(cases)}")
        for case_name, pred_img, tee_img in cases:
            case_dir = pred_img.parent
            if should_skip_case(case_dir, args.pred_name, args.tee_name, overwrite=args.overwrite):
                print(f"Skip case {case_name}: labels already exist (use --overwrite to relabel)")
                continue
            run_single_case(case_name, pred_img, tee_img)
        print("\nBatch labeling finished.")
        return

    image1 = Path(args.image1).expanduser().resolve()
    image2 = Path(args.image2).expanduser().resolve()
    if not image1.exists():
        raise FileNotFoundError(f"Image not found: {image1}")
    if not image2.exists():
        raise FileNotFoundError(f"Image not found: {image2}")
    run_single_case("single", image1, image2)


if __name__ == "__main__":
    main()
