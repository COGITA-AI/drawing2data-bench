from rfdetr import RFDETRLarge
from rfdetr.datasets.aug_configs import AUG_CONSERVATIVE

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


if __name__ == "__main__":
    model = RFDETRLarge(num_classes=9, resolution=704, device="cuda")

    model.train(dataset_dir="./dataset", 
                epochs=50,
                batch_size=8,
                grad_accum_steps=2,
                aug_config=DOCUMENT_SCAN_AUG,
                resume="output/last.ckpt",
                )

    model.export()