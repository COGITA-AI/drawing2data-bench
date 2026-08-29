import argparse
from rfdetr import RFDETRLarge
from rfdetr.datasets.aug_configs import AUG_CONSERVATIVE
from pathlib import Path

DOCUMENT_SCAN_AUG = {"Sequential": {"transforms": [
    {"Rotate": {"limit": 15, "border_mode": 0, "p": 0.4}},

    {"Perspective": {"scale": (0.02, 0.05), "keep_size": True, "p": 0.3}},

    {"OneOf": {
        "transforms": [
            {"RandomBrightnessContrast": {
                "brightness_limit": 0.15, "contrast_limit": 0.15, "p": 1.0}},
            {"RandomShadow": {
                "shadow_roi": (0, 0, 1, 1),
                "num_shadows_limit": (1, 2),
                "shadow_dimension": 5,
                "p": 1.0}},
            {"RandomSunFlare": {
                "flare_roi": (0, 0, 1, 0.5),
                "src_radius": 100,
                "p": 1.0}},
        ],
        "p": 0.5,
    }},

    {"OneOf": {
        "transforms": [
            {"GaussianBlur": {"blur_limit": (3, 5), "p": 1.0}},
            {"MotionBlur": {"blur_limit": (3, 7), "p": 1.0}},
            {"Defocus": {"radius": (1, 3), "p": 1.0}},
        ],
        "p": 0.3,
    }},

    {"OneOf": {
        "transforms": [
            {"GaussNoise": {"std_range": (0.02, 0.08), "p": 1.0}},
            {"ISONoise": {"color_shift": (0.01, 0.03),
                          "intensity": (0.1, 0.3), "p": 1.0}},
            {"ImageCompression": {"quality_range": (40, 80), "p": 1.0}},
        ],
        "p": 0.4,
    }},

    {"OneOf": {
        "transforms": [
            {"ColorJitter": {
                "brightness": 0.1, "contrast": 0.1,
                "saturation": 0.15, "hue": 0.03, "p": 1.0}},
            {"ToGray": {"p": 1.0}},
        ],
        "p": 0.35,
    }},

    {"Downscale": {"scale_range": (0.6, 0.9), "p": 0.15}},
], "p": 1.0}}


def parse_args():
    parser = argparse.ArgumentParser(description="Train RF-DETR Large on document dataset")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from (e.g. output/last.ckpt). "
             "If omitted, training starts from scratch.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./dataset",
        help="Path to the dataset directory (default: ./dataset)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (default: 2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.resume is not None and not Path(args.resume).exists():
        raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")

    model = RFDETRLarge(resolution=704, device="cuda")

    model.train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        aug_config=DOCUMENT_SCAN_AUG,
        resume=args.resume,
    )

    model.export()