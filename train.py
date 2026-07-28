"""Fine-tuning de modelos YOLOv8 para saúde da mulher — 3 datasets separados.

Uso:
    python train.py --dataset emotions  --epochs 30 --batch 8
    python train.py --dataset blood      --epochs 50 --batch 8
    python train.py --dataset instruments --epochs 30 --batch 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from src.models.yolo_detector import YOLODetector
from src.config import settings, resolve_device

DATASETS = {
    "emotions": "data/dataset/emotions/data.yaml",
    "blood": "data/dataset/blood/data.yaml",
    "instruments": "data/dataset/surgery_instruments/data.yaml",
}


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning YOLOv8 para saúde da mulher")
    parser.add_argument("--dataset", choices=["emotions", "blood", "instruments"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    yaml_path = DATASETS[args.dataset]
    if not Path(yaml_path).exists():
        logger.error(f"Dataset YAML não encontrado: {yaml_path}")
        return

    if args.dataset == "instruments":
        logger.info("Convertendo labels OBB → bbox...")
        import subprocess
        subprocess.run(["python", "convert_obb.py"], check=True)

    device = resolve_device(settings.device)
    logger.info(f"Treinando {args.dataset} | epochs={args.epochs} | device={device} | {yaml_path}")

    detector = YOLODetector(model_path="yolov8n.pt")
    detector.finetune(
        data_yaml=yaml_path,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        model_name=args.dataset,
    )


if __name__ == "__main__":
    main()
