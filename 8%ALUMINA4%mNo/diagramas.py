import math
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ternary

# ==========================================
# CONSTANTS & CONFIG
# ==========================================
SINGLE_COLUMN_MM = 85
DOUBLE_COLUMN_MM = 170
MM_TO_INCH = 1 / 25.4

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.titlesize": 10,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.0,
    "xtick.direction": "inout",
    "ytick.direction": "inout",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "svg.fonttype": "none",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
})

LABEL_MAP = {
    "PCTFEO": "FeO (%)",
    "PCTSIO2": "SiO$_2$ (%)",
    "PCTMGO": "MgO (%)",
    "PCTCAO": "CaO (%)",
}

AL2O3_PCT = 8.0
MNO_PCT = 4.0


def parse_thermocalc_exp(content):
    blocks = []
    current_block = None
    metadata = {"xscale": None, "yscale": None, "xtext": None, "ytext": None}

    number_pattern = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
    lines = content.split("\n")

    header_starts = (
        "PROLOG", "XSCALE", "YSCALE", "XTYPE", "YTYPE", "XLENGTH", "YLENGTH",
        "TITLE", "XTEXT", "YTEXT", "DATASET", "CHAR", "COLOR",
        "$ BLOCK", "$BLOCK", "$F0", "$E", "BLOCK", "BLOCKEND"
    )

    for line in lines:
        line = line.strip()
        if not line:
            if current_block and current_block["current_segment"]:
                current_block["segments"].append(np.array(current_block["current_segment"]))
                current_block["current_segment"] = []
            continue

        if line.startswith("XSCALE"):
            nums = [float(x) for x in number_pattern.findall(line)]
            if len(nums) >= 2:
                metadata["xscale"] = nums
        elif line.startswith("YSCALE"):
            nums = [float(x) for x in number_pattern.findall(line)]
            if len(nums) >= 2:
                metadata["yscale"] = nums
        elif line.startswith("XTEXT"):
            metadata["xtext"] = line
        elif line.startswith("YTEXT"):
            metadata["ytext"] = line

        if line.startswith("$ BLOCK") or line.startswith("$BLOCK"):
            if current_block:
                blocks.append(current_block)
            current_block = {"phases": [], "segments": [], "current_segment": []}

        elif line.startswith("$F0") or line.startswith("$E"):
            if current_block:
                phase_name = line.split(maxsplit=1)[1] if len(line.split()) > 1 else "Unknown"
                current_block["phases"].append(phase_name)

        elif line.startswith("BLOCKEND"):
            if current_block:
                if current_block["current_segment"]:
                    current_block["segments"].append(np.array(current_block["current_segment"]))
                blocks.append(current_block)
                current_block = None

        elif line.startswith("BLOCK") or line.startswith("$"):
            pass

        else:
            if current_block:
                if line.startswith(header_starts):
                    if current_block["current_segment"]:
                        current_block["segments"].append(np.array(current_block["current_segment"]))
                        current_block["current_segment"] = []
                    continue
                numbers = [float(x) for x in number_pattern.findall(line)]
                if len(numbers) >= 2:
                    x_val, y_val = numbers[0], numbers[1]

                    if "M" in line:
                        if current_block["current_segment"]:
                            current_block["segments"].append(np.array(current_block["current_segment"]))
                            current_block["current_segment"] = []
                        current_block["current_segment"].append([x_val, y_val])
                    else:
                        current_block["current_segment"].append([x_val, y_val])
                else:
                    if current_block["current_segment"]:
                        current_block["segments"].append(np.array(current_block["current_segment"]))
                        current_block["current_segment"] = []

    if current_block:
        if current_block["current_segment"]:
            current_block["segments"].append(np.array(current_block["current_segment"]))
        blocks.append(current_block)

    return blocks, metadata


def detect_scale(blocks):
    max_val = 0.0
    for block in blocks:
        for segment in block["segments"]:
            if len(segment) == 0:
                continue
            max_val = max(max_val, float(np.max(segment)))
    return 100.0 if max_val <= 1.0 else 1.0


def extract_label(text_line):
    if not text_line:
        return None
    match = re.search(r"(PCT[A-Z0-9]+|W\([A-Z0-9]+\))", text_line)
    if not match:
        return None
    key = match.group(1)
    if key.startswith("W(") and key.endswith(")"):
        key = f"PCT{key[2:-1]}"
    return LABEL_MAP.get(key, key)


def plot_phase_diagram(exp_path, figures_dir):
    with open(exp_path, "r", encoding="latin-1") as f:
        content = f.read()

    blocks, metadata = parse_thermocalc_exp(content)
    if not blocks:
        return None

    scale = detect_scale(blocks)
    x_label = extract_label(metadata.get("xtext")) or "FeO (%)"
    y_label = extract_label(metadata.get("ytext")) or "SiO$_2$ (%)"

    base_name = os.path.basename(exp_path)
    cao_pct = None
    match = re.search(r"(\d+)", base_name)
    if match:
        cao_pct = float(match.group(1))

    if cao_pct is None:
        print(f"No se pudo extraer CaO desde {base_name}, se omite.")
        return None

    total_tri = 100.0 - AL2O3_PCT - MNO_PCT - cao_pct
    if total_tri <= 0:
        print(f"Total ternario invalido para {base_name}, se omite.")
        return None

    width = DOUBLE_COLUMN_MM * MM_TO_INCH
    height = width
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("white")
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=total_tri)

    tax.boundary(linewidth=1.2, axes_colors={"l": "black", "r": "black", "b": "black"})
    tax.gridlines(multiple=10, linewidth=0.6, color="#c7c7c7", linestyle="--")
    tax.ticks(axis="lbr", multiple=10, linewidth=0.9, offset=0.02, fontsize=9)

    # Ejes solicitados: izquierdo SiO2, base FeO, derecho MgO
    tax.left_axis_label("SiO$_2$ (%)", fontsize=10, offset=0.14)
    tax.right_axis_label("MgO (%)", fontsize=10, offset=0.14)
    tax.bottom_axis_label("FeO (%)", fontsize=10, offset=0.14)

    colors = [
        "#0b1f3a", "#2a6f97", "#52b788", "#ef476f", "#f4a261",
        "#8338ec", "#3d405b", "#8d99ae", "#264653",
    ]

    for block in blocks:
        phases_set = set(block["phases"])
        color_seed = sum(map(ord, "".join(sorted(phases_set))))
        color = colors[color_seed % len(colors)]

        for segment in block["segments"]:
            if len(segment) == 0:
                continue
            seg_scaled = segment * scale
            coords = []
            for feo, sio2 in seg_scaled:
                mgo = total_tri - feo - sio2
                if mgo < 0:
                    if len(coords) >= 2:
                        tax.plot(coords, color=color, linewidth=0.8)
                    coords = []
                    continue
                coords.append((sio2, mgo, feo))
            if len(coords) >= 2:
                tax.plot(coords, color=color, linewidth=1.1)

    tax.set_title(
        f"Diagrama ternario (Al$_2$O$_3$ 8%, MnO 4%) - {base_name.replace('.exp', '')}",
        fontsize=11,
        pad=14,
    )
    tax.clear_matplotlib_ticks()
    plt.tight_layout()
    output_path = os.path.join(
        figures_dir,
        f"diagrama_fases_{os.path.basename(exp_path).replace('.exp', '')}.png",
    )
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_diagramas():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    exp_files = sorted(
        f for f in os.listdir(script_dir) if f.lower().endswith(".exp")
    )
    if not exp_files:
        print("No se encontraron archivos .exp en la carpeta del script.")
        return

    for filename in exp_files:
        exp_path = os.path.join(script_dir, filename)
        output = plot_phase_diagram(exp_path, figures_dir)
        if output:
            print(f"Guardado: {output}")


if __name__ == "__main__":
    run_diagramas()
