"""Converte labels OBB (8 coordenadas) para bbox YOLO padrão (x_center, y_center, width, height).
Usado pelo dataset surgery_instruments do Roboflow.
"""

from pathlib import Path
import shutil

SRC = Path("data/dataset/surgery_instruments")


def obb_to_bbox(line: str) -> str:
    parts = line.strip().split()
    if len(parts) == 5:
        return line.strip()
    if len(parts) != 9:
        return ""
    cls_id = parts[0]
    coords = [float(x) for x in parts[1:]]
    xs = [coords[i] for i in range(0, 8, 2)]
    ys = [coords[i] for i in range(1, 8, 2)]
    x_c = (min(xs) + max(xs)) / 2
    y_c = (min(ys) + max(ys)) / 2
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"


def main():
    for split in ["train", "valid", "test"]:
        lbl_dir = SRC / split / "labels"
        if not lbl_dir.exists():
            continue
        converted = 0
        for lbl in lbl_dir.glob("*.txt"):
            lines = []
            for line in lbl.read_text().strip().splitlines():
                bbox_line = obb_to_bbox(line)
                if bbox_line:
                    lines.append(bbox_line)
            if lines:
                lbl.write_text("\n".join(lines))
                converted += 1
        print(f"{split}: {converted} labels convertidas (OBB → bbox)")

    print("Pronto. Dataset surgery_instruments agora usa bbox padrão.")


if __name__ == "__main__":
    main()
