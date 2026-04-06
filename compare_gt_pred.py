"""
Generate side-by-side Ground Truth vs Prediction comparison figures
for selected test-set samples.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "newdata_yolo26_960run_100_32" / "weights" / "best.pt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "test_result" / "gt_pred_comparison"

CLASS_NAMES = {0: "vehicle", 1: "person", 2: "traffic light", 3: "traffic sign"}
CLASS_COLORS = {
    0: (0.0, 0.8, 0.0),   # green
    1: (0.0, 0.4, 1.0),   # blue
    2: (1.0, 0.0, 0.0),   # red
    3: (1.0, 0.65, 0.0),  # orange
}

DEFAULT_TEST_IMAGES = [
    "cabea010-6882cf41.jpg",
    "cabc9045-1b8282ba.jpg",
    "fa2f1888-5fbbf058.jpg",
    "e25d821e-3b469b66.jpg",
    "f90bf99a-6b15cca2.jpg",
]


def parse_args():
    parser = ArgumentParser(
        description="Generate side-by-side ground-truth versus prediction comparison figures."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained YOLO weights file.",
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        required=True,
        help="Directory containing test images.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        required=True,
        help="Directory containing YOLO-format test labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where comparison figures will be saved.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Ultralytics device string, for example 'mps', 'cpu', or '0'.",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=DEFAULT_TEST_IMAGES,
        help="Specific image filenames to compare.",
    )
    return parser.parse_args()


def parse_yolo_labels(label_path: Path, img_w: int, img_h: int):
    """Parse YOLO labels and convert them to pixel coordinates."""
    boxes = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            bw = w * img_w
            bh = h * img_h
            boxes.append((cls_id, x1, y1, bw, bh))
    return boxes


def draw_boxes(ax, boxes, title):
    """Draw bounding boxes on a matplotlib axis."""
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    for cls_id, x, y, w, h in boxes:
        color = CLASS_COLORS.get(cls_id, (0.5, 0.5, 0.5))
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1.5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        label = CLASS_NAMES.get(cls_id, str(cls_id))
        ax.text(
            x,
            max(y - 2, 0),
            label,
            fontsize=6,
            color="white",
            bbox=dict(facecolor=color, alpha=0.7, pad=0.3, edgecolor="none"),
        )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model_path))

    for idx, img_name in enumerate(args.images, start=1):
        img_path = args.img_dir / img_name
        label_path = args.label_dir / img_name.replace(".jpg", ".txt")

        if not img_path.exists():
            print(f"[SKIP] Missing image: {img_path}")
            continue
        if not label_path.exists():
            print(f"[SKIP] Missing label: {label_path}")
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size

        gt_boxes = parse_yolo_labels(label_path, img_w, img_h)

        results = model(str(img_path), imgsz=args.imgsz, device=args.device, verbose=False)[0]
        pred_boxes = []
        for box in results.boxes:
            cls_id = int(box.cls)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pred_boxes.append((cls_id, x1, y1, x2 - x1, y2 - y1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        ax1.imshow(img)
        draw_boxes(ax1, gt_boxes, "Ground Truth")
        ax2.imshow(img)
        draw_boxes(ax2, pred_boxes, "Prediction")

        legend_handles = [
            patches.Patch(facecolor=CLASS_COLORS[key], label=CLASS_NAMES[key])
            for key in sorted(CLASS_NAMES)
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=11, frameon=True)
        fig.suptitle(img_name, fontsize=12, y=0.98)
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])

        out_path = args.output_dir / f"comparison_{idx}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
