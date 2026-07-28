"""Gera dataset sintético de imagens para treino do YOLOv8 customizado.

Cria imagens 640x640 com formas geométricas simulando objetos cirúrgicos,
sangramento e expressões faciais, com labels YOLO correspondentes.

Uso:
    python generate_dataset.py             # 100 imagens treino, 30 val
    python generate_dataset.py --train 200 --val 50
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CLASSES = {
    0: "bisturi",
    1: "tesoura_cirurgica",
    2: "pinça_hemostatica",
    3: "afastador_cirurgico",
    4: "sutura",
    5: "compressa",
    6: "sangramento_normal",
    7: "sangramento_anomalo",
    8: "utero",
    9: "ovarios",
    10: "mamas",
    11: "area_critica",
    12: "objeto_suspeito",
    13: "expressao_desconforto",
    14: "expressao_medo",
}

COLORS = {
    0: (192, 192, 192),
    1: (180, 180, 180),
    2: (160, 160, 180),
    3: (170, 170, 190),
    4: (200, 180, 150),
    5: (220, 220, 220),
    6: (180, 40, 40),
    7: (220, 20, 20),
    8: (200, 140, 160),
    9: (200, 160, 180),
    10: (210, 150, 170),
    11: (255, 200, 50),
    12: (100, 100, 100),
    13: (255, 180, 100),
    14: (255, 120, 80),
}


def draw_bisturi(draw, x, y, w, h, color):
    draw.rectangle([x, y, x + w, y + h // 2], fill=color)
    draw.rectangle([x + w // 3, y + h // 2, x + 2 * w // 3, y + h], fill=color)


def draw_tesoura(draw, x, y, w, h, color):
    cx, cy = x + w // 2, y + h // 2
    draw.line([cx, y, cx - w // 4, y + h // 2], fill=color, width=3)
    draw.line([cx, y, cx + w // 4, y + h // 2], fill=color, width=3)
    draw.line([cx - w // 4, y + h // 2, cx + w // 4, y + h // 2], fill=color, width=2)
    draw.ellipse([cx - w // 6, y + h // 2 - 2, cx + w // 6, y + h // 2 + 8], fill=color)


def draw_pinca(draw, x, y, w, h, color):
    draw.line([x + w // 2, y, x + w // 2, y + h], fill=color, width=4)
    draw.rectangle([x + w // 2 - 2, y + h - 10, x + w // 2 + 2, y + h], fill=color)


def draw_afastador(draw, x, y, w, h, color):
    draw.arc([x, y, x + w, y + h * 2], 180, 360, fill=color, width=5)
    draw.rectangle([x + w // 2 - 3, y + h // 2, x + w // 2 + 3, y + h], fill=color)


def draw_sutura(draw, x, y, w, h, color):
    for i in range(0, w, 6):
        draw.line([x + i, y + h // 2, x + i + 3, y + h // 2 + (3 if i % 12 == 0 else -3)], fill=color, width=1)


def draw_compressa(draw, x, y, w, h, color):
    for i in range(0, w, 12):
        for j in range(0, h, 12):
            draw.line([x + i, y + j, x + i + 10, y + j + 10], fill=color, width=1)


def draw_sangramento(draw, x, y, w, h, color, anomalo=False):
    blobs = 15 if anomalo else 5
    for _ in range(blobs):
        bx = x + random.randint(0, w)
        by = y + random.randint(0, h)
        br = random.randint(3, 10 if anomalo else 5)
        draw.ellipse([bx - br, by - br, bx + br, by + br], fill=color)
    if anomalo:
        draw.ellipse([x + w // 2 - w // 4, y + h // 2 - h // 4, x + w // 2 + w // 4, y + h // 2 + h // 4],
                     fill=(180, 0, 0))


def draw_utero(draw, x, y, w, h, color):
    cx, cy = x + w // 2, y + h // 2
    draw.ellipse([cx - w // 3, cy - h // 2, cx + w // 3, cy + h // 2], fill=color, outline=(150, 100, 120), width=2)
    draw.ellipse([cx - w // 6, cy - h // 2 + 5, cx + w // 6, cy + h // 3], fill=(250, 200, 210))


def draw_ovarios(draw, x, y, w, h, color):
    r = min(w, h) // 2
    draw.ellipse([x + w // 2 - r, y + h // 2 - r, x + w // 2 + r, y + h // 2 + r], fill=color, outline=(180, 140, 160))


def draw_mamas(draw, x, y, w, h, color):
    draw.ellipse([x, y + h // 3, x + w, y + h], fill=color, outline=(190, 130, 150))
    draw.ellipse([x + w // 4, y, x + 3 * w // 4, y + h // 3], fill=(255, 180, 190))


def draw_area_critica(draw, x, y, w, h, color):
    for _ in range(5):
        rx = x + random.randint(0, w - 10)
        ry = y + random.randint(0, h - 10)
        draw.rectangle([rx, ry, rx + 10, ry + 10], fill=color, outline=(255, 100, 0), width=2)
    draw.text((x + 2, y + 2), "!", fill=(255, 0, 0))


def draw_objeto_suspeito(draw, x, y, w, h, color):
    cx, cy = x + w // 2, y + h // 2
    for angle in range(0, 360, 30):
        rad = angle * 3.14159 / 180
        ex = cx + int(w // 3 * np.cos(rad))
        ey = cy + int(h // 3 * np.sin(rad))
        draw.line([cx, cy, ex, ey], fill=color, width=2)
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(200, 0, 0))


def draw_expressao(draw, x, y, w, h, color, medo=False):
    eye_y = y + h // 3
    eye_spacing = w // 3
    eye_r = w // 6
    m = 8 if medo else 3
    draw.ellipse([x + eye_spacing - eye_r, eye_y - eye_r, x + eye_spacing + eye_r, eye_y + eye_r], fill=(0, 0, 0))
    draw.ellipse([x + 2 * eye_spacing - eye_r, eye_y - eye_r, x + 2 * eye_spacing + eye_r, eye_y + eye_r], fill=(0, 0, 0))
    mouth_y = y + 2 * h // 3
    draw.arc([x + eye_spacing, mouth_y - m, x + 2 * eye_spacing, mouth_y + m], 0 if medo else 180, 180 if medo else 360, fill=color, width=2)


DRAW_FUNCS = {
    0: draw_bisturi,
    1: draw_tesoura,
    2: draw_pinca,
    3: draw_afastador,
    4: draw_sutura,
    5: draw_compressa,
    6: draw_sangramento,
    7: lambda d, x, y, w, h, c: draw_sangramento(d, x, y, w, h, c, anomalo=True),
    8: draw_utero,
    9: draw_ovarios,
    10: draw_mamas,
    11: draw_area_critica,
    12: draw_objeto_suspeito,
    13: draw_expressao,
    14: lambda d, x, y, w, h, c: draw_expressao(d, x, y, w, h, c, medo=True),
}


def generate_image(size=640, num_objects=None):
    if num_objects is None:
        num_objects = random.randint(2, 6)

    img = Image.new("RGB", (size, size), color=(40, 40, 50))
    draw = ImageDraw.Draw(img)

    labels = []
    chosen = random.sample(list(CLASSES.keys()), min(num_objects, len(CLASSES)))

    for cls_id in chosen:
        w = random.randint(40, 140)
        h = random.randint(40, 140)
        x = random.randint(10, size - w - 10)
        y = random.randint(10, size - h - 10)

        color = COLORS[cls_id]
        draw_func = DRAW_FUNCS[cls_id]
        draw_func(draw, x, y, w, h, color)

        x_center = (x + w / 2) / size
        y_center = (y + h / 2) / size
        width_norm = w / size
        height_norm = h / size
        labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")

    return img, labels


def generate_split(dataset_root: Path, split_name: str, num_images: int, size=640):
    img_dir = dataset_root / "images" / split_name
    lbl_dir = dataset_root / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        img, labels = generate_image(size=size)
        name = f"{split_name}_{i:04d}"
        img.save(img_dir / f"{name}.jpg")
        with open(lbl_dir / f"{name}.txt", "w") as f:
            f.write("\n".join(labels))

    print(f"  {split_name}: {num_images} imagens → {img_dir}")


def main():
    parser = argparse.ArgumentParser(description="Gerador de dataset sintético para YOLOv8")
    parser.add_argument("--train", type=int, default=100)
    parser.add_argument("--val", type=int, default=30)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--out", type=str, default="data/dataset")
    args = parser.parse_args()

    base = Path(args.out)
    base.mkdir(parents=True, exist_ok=True)
    print(f"Gerando dataset sintético em {base.resolve()}/")

    generate_split(base, "train", args.train, args.size)
    generate_split(base, "val", args.val, args.size)
    if args.test:
        generate_split(base, "test", args.test, args.size)

    train_count = len(list((base / "images" / "train").glob("*.jpg")))
    val_count = len(list((base / "images" / "val").glob("*.jpg")))
    print(f"\nDataset pronto: {train_count} treino + {val_count} validação")
    print(f"Execute: python train.py --config models/yolov8_config.yaml --epochs 30 --batch 8")


if __name__ == "__main__":
    main()
