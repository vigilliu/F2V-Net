#!/usr/bin/env python3
"""
Interactive two-point annotation for:
- pred_slice.png
- tee_random_mid_256.png

For each image, click 2 points in order.
The coordinates are saved in the same folder as the image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import tkinter as tk


Point = Tuple[float, float]


class TwoPointMarker:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path
        self.points: List[Point] = []
        self.done = False
        self.cancelled = False

        self.root = tk.Tk()
        self.root.title(f"Mark 2 points: {self.image_path.name}")

        self.photo = tk.PhotoImage(file=str(self.image_path))
        self.canvas = tk.Canvas(
            self.root,
            width=self.photo.width(),
            height=self.photo.height(),
            highlightthickness=0,
        )
        self.canvas.pack()
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.label = tk.Label(self.root, text="Click point 1/2", anchor="w")
        self.label.pack(fill="x")

        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<BackSpace>", self.on_undo)
        self.root.bind("<Escape>", self.on_cancel)
        self.root.protocol("WM_DELETE_WINDOW", self.on_cancel)

    def on_click(self, event: tk.Event) -> None:
        if self.done:
            return

        x, y = float(event.x), float(event.y)
        self.points.append((x, y))

        r = 4
        color = "red" if len(self.points) == 1 else "yellow"
        self.canvas.create_oval(x - r, y - r, x + r, y + r, outline=color, fill=color, width=2)
        self.canvas.create_text(x + 8, y - 8, text=f"P{len(self.points)}", fill=color, anchor=tk.W)

        if len(self.points) < 2:
            self.label.config(text="Click point 2/2")
            return

        self.done = True
        self.label.config(text="Done, closing...")
        self.root.after(300, self.root.destroy)

    def on_undo(self, _event: tk.Event) -> None:
        if self.done or not self.points:
            return
        self.points.pop()
        self._redraw()
        next_id = len(self.points) + 1
        self.label.config(text=f"Click point {next_id}/2")

    def _redraw(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        for idx, (x, y) in enumerate(self.points, start=1):
            r = 4
            color = "red" if idx == 1 else "yellow"
            self.canvas.create_oval(x - r, y - r, x + r, y + r, outline=color, fill=color, width=2)
            self.canvas.create_text(x + 8, y - 8, text=f"P{idx}", fill=color, anchor=tk.W)

    def on_cancel(self, _event: tk.Event | None = None) -> None:
        self.cancelled = True
        self.root.destroy()

    def run(self) -> List[Point]:
        self.root.mainloop()
        if self.cancelled:
            raise RuntimeError(f"Annotation cancelled: {self.image_path}")
        if len(self.points) != 2:
            raise RuntimeError(f"Expected 2 points, got {len(self.points)}: {self.image_path}")
        return self.points


def save_points(image_path: Path, points: List[Point]) -> Path:
    output_path = image_path.parent / f"{image_path.stem}_points.json"
    payload = {
        "image_name": image_path.name,
        "point_order": ["P1", "P2"],
        "points": [
            {"x": float(points[0][0]), "y": float(points[0][1])},
            {"x": float(points[1][0]), "y": float(points[1][1])},
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_path


def collect_cases(root_dir: Path, pred_name: str, tee_name: str) -> List[tuple[Path, Path]]:
    pairs: List[tuple[Path, Path]] = []
    for case_dir in sorted(root_dir.iterdir(), key=lambda p: p.name):
        if not case_dir.is_dir():
            continue
        pred_img = case_dir / pred_name
        tee_img = case_dir / tee_name
        if pred_img.exists() and tee_img.exists():
            pairs.append((pred_img, tee_img))
    return pairs


def annotate_image(image_path: Path, overwrite: bool) -> None:
    out_json = image_path.parent / f"{image_path.stem}_points.json"
    if out_json.exists() and not overwrite:
        print(f"Skip (exists): {out_json}")
        return

    print(f"\nAnnotating: {image_path}")
    print("Hotkeys: Backspace=undo, Esc=cancel")
    marker = TwoPointMarker(image_path)
    points = marker.run()
    saved_path = save_points(image_path, points)
    print(f"Saved: {saved_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively mark 2 points for pred_slice.png and tee_random_mid_256.png."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Dataset root directory containing case subfolders.",
    )
    parser.add_argument("--pred-name", type=str, default="pred_slice.png", help="Pred image filename.")
    parser.add_argument("--tee-name", type=str, default="tee_random_mid_256.png", help="TEE image filename.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_points.json files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    cases = collect_cases(root, args.pred_name, args.tee_name)
    if not cases:
        raise RuntimeError(
            f"No valid subfolders found under {root} with both {args.pred_name} and {args.tee_name}."
        )

    print(f"Found {len(cases)} case(s) under: {root}")
    for pred_img, tee_img in cases:
        annotate_image(pred_img, overwrite=args.overwrite)
        annotate_image(tee_img, overwrite=args.overwrite)
    print("\nAll done.")


if __name__ == "__main__":
    main()
