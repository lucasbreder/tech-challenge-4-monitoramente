"""Unifica 3 datasets Roboflow em um único dataset YOLOv8 para treino.

Datasets:
  - blood (1 classe: sangramento)
  - surgery_instruments (21 classes, OBB → bbox)
  - emotions (8 classes: medo, tristeza, raiva, etc.)

Gera data/dataset/unified/ com classes offsetadas e um data.yaml unificado.
"""

from pathlib import Path
import shutil

BASE = Path("data/dataset")
UNIFIED = BASE / "unified"


def obb_to_bbox(parts: list[str]) -> str:
    """Converte OBB (8 coords) para bbox YOLO (x_center, y_center, width, height)."""
    cls_id = parts[0]
    coords = [float(x) for x in parts[1:]]
    xs = [coords[0], coords[2], coords[4], coords[6]]
    ys = [coords[1], coords[3], coords[5], coords[7]]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_c = (x_min + x_max) / 2
    y_c = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"


def convert_labels(src_dir: Path, dst_dir: Path, offset: int) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for lbl in src_dir.glob("*.txt"):
        lines = []
        for line in lbl.read_text().strip().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            old_cls = int(parts[0])
            new_cls = old_cls + offset

            if len(parts) == 9:
                parts[0] = str(new_cls)
                lines.append(obb_to_bbox(parts))
            elif len(parts) == 5:
                parts[0] = str(new_cls)
                lines.append(" ".join(parts))

        if lines:
            (dst_dir / lbl.name).write_text("\n".join(lines))
            count += 1
    return count


def symlink_images(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in src_dir.glob("*.jpg"):
        dst = dst_dir / img.name
        if not dst.exists():
            dst.symlink_to(img.resolve())
        count += 1
    return count


DATASETS = [
    {
        "name": "blood",
        "classes": ["sangramento"],
        "offset": 0,
    },
    {
        "name": "surgery_instruments",
        "classes": [
            "bisturi", "tesoura_cirurgica", "pinça_hemostatica", "afastador_cirurgico",
            "sutura", "compressa", "cabo_bisturi", "clamp",
            "aspirador", "eletrodo", "pinça_disseccao",
            "porta_agulha", "tesoura_curva", "pinça_kelly",
            "pinça_mosquito", "afastador_gelpi", "pinça_allis",
            "pinça_babcock", "pinça_foerster", "tesoura_mayo",
            "pinça_adson",
        ],
        "offset": 1,
    },
    {
        "name": "emotions",
        "classes": [
            "expressao_raiva", "expressao_satisfacao", "expressao_nojo",
            "expressao_medo", "expressao_feliz", "expressao_neutra",
            "expressao_triste", "expressao_surpresa",
        ],
        "offset": 22,
    },
]


def main():
    if UNIFIED.exists():
        print(f"Removendo dataset unificado existente em {UNIFIED}...")
        shutil.rmtree(UNIFIED)

    all_classes = []
    for ds in DATASETS:
        all_classes.extend(ds["classes"])

    for split in ["train", "valid", "test"]:
        n_total_img, n_total_lbl = 0, 0
        for ds in DATASETS:
            src = BASE / ds["name"]
            src_img = src / split / "images"
            src_lbl = src / split / "labels"
            if not src_img.exists():
                continue

            dst_img = UNIFIED / "images" / split
            dst_lbl = UNIFIED / "labels" / split

            n_total_img += symlink_images(src_img, dst_img)
            n_total_lbl += convert_labels(src_lbl, dst_lbl, ds["offset"])

        print(f"{split}: {n_total_img} imagens, {n_total_lbl} labels")

    names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(all_classes))
    data_yaml = UNIFIED / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(f"""path: {UNIFIED.resolve()}
train: images/train
val: images/valid
test: images/test

nc: {len(all_classes)}
names:
{names_yaml}
""")

    print(f"\nTotal: {len(all_classes)} classes unificadas:")
    for i, name in enumerate(all_classes):
        print(f"  {i}: {name}")
    print(f"\nDataset unificado: {UNIFIED}")
    print(f"\nTreine com:")
    print(f"  python train.py --config {data_yaml} --epochs 30 --batch 8")


if __name__ == "__main__":
    main()
