import argparse
from pathlib import Path

from rfdetr import RFDETRLarge
import supervision as sv
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Run RF-DETR Large inference on an image")
    parser.add_argument(
        "image",
        type=str,
        help="Path to the input image (e.g. image.png)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="annotated.png",
        help="Path to save the annotated image (default: annotated.png)",
    )
    parser.add_argument(
        "-w", "--weights",
        type=str,
        default="output/checkpoint_best_total.pth",
        help="Path to model weights (default: output/checkpoint_best_total.pth)",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    image = Image.open(image_path)

    model = RFDETRLarge(
        resolution=1056,
        pretrain_weights=str(weights_path),
        device=args.device,
    )
    model.optimize_for_inference()

    detections = model.predict(image, threshold=args.threshold)

    color = sv.ColorPalette.from_hex([
        "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
        "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
    ])
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

    bbox_annotator = sv.BoxAnnotator(color=color, thickness=thickness)

    annotated_image = image.copy()
    annotated_image = bbox_annotator.annotate(annotated_image, detections)
    annotated_image.save(args.output)

    print(f"Detected {len(detections)} object(s). Saved annotated image to: {args.output}")


if __name__ == "__main__":
    main()