"""Script de fine-tuning do YOLOv8 para o dataset customizado de saúde da mulher.

Uso:
    python train.py                          # usa config padrão (models/yolov8_config.yaml)
    python train.py --config meu_dataset.yaml --epochs 100 --batch 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from src.models.yolo_detector import YOLODetector
from src.config import settings


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning YOLOv8 para saúde da mulher")
    parser.add_argument(
        "--config", type=str, default=settings.model.yolo_config_path,
        help="Caminho do YAML de configuração do dataset",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Épocas de treino")
    parser.add_argument("--img-size", type=int, default=640, help="Resolução da imagem")
    parser.add_argument("--batch", type=int, default=16, help="Tamanho do batch")
    parser.add_argument("--device", type=str, default=settings.device, help="cpu/cuda/mps")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Arquivo de configuração não encontrado: {config_path}")
        logger.info("Certifique-se de que o dataset está em data/dataset/ conforme o YAML")
        return

    logger.info(f"Iniciando fine-tuning com config: {config_path}")
    logger.info(f"Parâmetros: epochs={args.epochs}, imgsz={args.img_size}, batch={args.batch}, device={args.device}")

    detector = YOLODetector(model_path="yolov8n.pt")
    detector.finetune(
        data_yaml=str(config_path),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
    )

    logger.info("Treino concluído! Modelo salvo em models/yolov8_custom.pt")


if __name__ == "__main__":
    main()
