# flake8: noqa
import math
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

# ==========================================
# CONSTANTS & CONFIG
# ==========================================
SINGLE_COLUMN_MM = 85
DOUBLE_COLUMN_MM = 170
MM_TO_INCH = 1 / 25.4

AL2O3_PCT = 8.0
MNO_PCT = 4.0

FILES_TO_PROCESS = ["30.exp", "35.exp", "40.exp"]
FILE_COLORS = {
    "30.exp": "black",
    "35.exp": "blue",
    "40.exp": "green",
}
EXCLUDED_BLOCK_IDS_40 = {4467,5111,5576,5954,1611,1466}
EXCLUDED_PHASE_LABELS_BLUE = {
    "Liquido#1 + Liquido#3 + Halite#1 + Halite#2",
    "Liquido#1 + Liquido#2 + Halite#1 + Halite#2",
}

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

# ==========================================
# HELPERS
# ==========================================

def puntos_EAF_completos():
    """
    EAF slag literature data.
    """
    return [
        {"FeO":25.0, "SiO2":14.0, "CaO":39.0, "MgO":14.0, "Al2O3":5.00, "MnO":3.00, "label":"Iran"},
        {"FeO":42.4, "SiO2":20.3, "CaO":17.0, "MgO":8.00,  "Al2O3":7.30, "MnO":5.00, "label":"India"},
        {"FeO":33.2, "SiO2":10.0, "CaO":31.6, "MgO":8.00,  "Al2O3":10.18,"MnO":7.00, "label":"China"},
        {"FeO":33.3, "SiO2":19.3, "CaO":28.7, "MgO":3.07, "Al2O3":9.40, "MnO":4.20, "label":"Malaysia"},
        {"FeO":36.8, "SiO2":13.1, "CaO":35.5, "MgO":5.03, "Al2O3":5.51, "MnO":3.20, "label":"Egypt"},
        {"FeO":31.7, "SiO2":21.7, "CaO":32.3, "MgO":2.60, "Al2O3":8.83, "MnO":2.90, "label":"Malaysia"},
        {"FeO":37.5, "SiO2":9.71, "CaO":38.0, "MgO":2.17, "Al2O3":8.21, "MnO":4.40, "label":"Italy"},
        {"FeO":43.4, "SiO2":26.4, "CaO":18.1, "MgO":1.86, "Al2O3":4.84, "MnO":5.40, "label":"Malaysia-CarbonSteel"},
        {"FeO":33.3, "SiO2":20.8, "CaO":30.5, "MgO":2.06, "Al2O3":9.19, "MnO":3.80, "label":"Malaysia"},
        {"FeO":35.0, "SiO2":14.0, "CaO":29.0, "MgO":5.00, "Al2O3":12.0, "MnO":5.00, "label":"Italy"},
        {"FeO":26.8, "SiO2":19.1, "CaO":34.9, "MgO":2.50, "Al2O3":13.7, "MnO":3.00, "label":"Spain-CarbonSteel"},
        {"FeO":22.3, "SiO2":20.3, "CaO":38.0, "MgO":3.00, "Al2O3":12.2, "MnO":4.20, "label":"Spain"},
        {"FeO":22.0, "SiO2":21.4, "CaO":38.5, "MgO":4.89, "Al2O3":9.60, "MnO":3.60, "label":"Malaysia"},
        {"FeO":34.7, "SiO2":16.3, "CaO":30.1, "MgO":6.86, "Al2O3":8.31, "MnO":3.70, "label":"Vietnam-CarbonSteel"},
        {"FeO":7.54, "SiO2":27.8, "CaO":46.6, "MgO":7.35, "Al2O3":2.74, "MnO":4.00, "label":"China-StainlessSteel"},
        {"FeO":24.5, "SiO2":20.9, "CaO":36.8, "MgO":3.20, "Al2O3":12.1, "MnO":2.50, "label":"Spain"},
        {"FeO":28.6, "SiO2":18.1, "CaO":32.7, "MgO":5.80, "Al2O3":5.88, "MnO":4.90, "label":"Malaysia"},
        {"FeO":43.0, "SiO2":10.8, "CaO":33.1, "MgO":1.65, "Al2O3":6.86, "MnO":4.60, "label":"Malaysia"},
        {"FeO":27.3, "SiO2":17.3, "CaO":38.4, "MgO":5.39, "Al2O3":4.67, "MnO":3.90, "label":"Malaysia"},
        {"FeO":25.9, "SiO2":19.5, "CaO":40.5, "MgO":4.25, "Al2O3":4.88, "MnO":3.00, "label":"Iran"},
    ]

def parse_thermocalc_exp(content):
    blocks = []
    current_block = None
    metadata: dict[str, list[float] | None] = {"xscale": None, "yscale": None}

    number_pattern = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
    block_id_pattern = re.compile(r"\$\s*BLOCK\s*#(\d+)", re.IGNORECASE)
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

        if line.startswith("$ BLOCK") or line.startswith("$BLOCK"):
            if current_block:
                blocks.append(current_block)
            block_id = None
            match = block_id_pattern.search(line)
            if match:
                block_id = int(match.group(1))
            current_block = {
                "phases": [],
                "segments": [],
                "current_segment": [],
                "block_id": block_id,
            }

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


def detect_scale(blocks, metadata):
    if metadata.get("xscale"):
        x_max = metadata["xscale"][1]
        if x_max <= 1.0:
            return 100.0

    max_val = 0.0
    for block in blocks:
        for segment in block["segments"]:
            if len(segment) == 0:
                continue
            max_val = max(max_val, float(np.max(segment)))
    return 100.0 if max_val <= 1.0 else 1.0


def ternary_to_xy(feo, sio2, mgo):
    # Vertices: left=MgO, right=FeO, top=SiO2
    total = feo + sio2 + mgo
    if total <= 0:
        return None
    feo /= total
    sio2 /= total
    mgo /= total
    x = feo + 0.5 * sio2
    y = (math.sqrt(3.0) / 2.0) * sio2
    return x, y


def extract_cao_from_filename(filename):
    match = re.search(r"(\d+)", filename)
    if not match:
        raise ValueError(f"No se pudo extraer CaO desde {filename}")
    return float(match.group(1))

def warm_shade(base_color, t):
    rgb = np.array(mcolors.to_rgb(base_color))
    warm = np.array([0.85, 0.35, 0.15])
    warm_mix = 0.15 + 0.35 * t
    color = (1.0 - warm_mix) * rgb + warm_mix * warm
    darken = 1.0 - (0.15 + 0.35 * t)
    return tuple(np.clip(color * darken, 0.0, 1.0))

def format_phase_label(phases):
    seen: list[str] = []
    for p in phases:
        if not isinstance(p, str) or not p:
            continue
        if p not in seen:
            seen.append(p)
    replacements = {
        "IONIC_LIQ#1": "Liquido#1",
        "IONIC_LIQ#2": "Liquido#2",
        "IONIC_LIQ#3": "Liquido#3",
        "HALITE#1": "Halite#1",
        "HALITE#2": "Halite#2",
    }
    pretty: list[str] = []
    for p in seen:
        label = replacements.get(p, p)
        pretty.append(label.replace("_", " "))
    return " + ".join(pretty)


def draw_ternary_grid(ax, step=10, color="0.85", linewidth=0.6):
    for val in range(step, 100, step):
        # Constant SiO2 (parallel to base)
        p1 = ternary_to_xy(0.0, val, 100.0 - val)
        p2 = ternary_to_xy(100.0 - val, val, 0.0)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth)

        # Constant FeO (parallel to left edge)
        p1 = ternary_to_xy(val, 0.0, 100.0 - val)
        p2 = ternary_to_xy(val, 100.0 - val, 0.0)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth)

        # Constant MgO (parallel to right edge)
        p1 = ternary_to_xy(0.0, 100.0 - val, val)
        p2 = ternary_to_xy(100.0 - val, 0.0, val)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth)


def draw_ternary_ticks(ax, step=10):
    tick_len = 0.015
    font_size = 8
    for val in range(step, 100, step):
        # FeO ticks along base (SiO2=0)
        x, y = ternary_to_xy(val, 0.0, 100.0 - val)
        ax.plot([x, x], [y, y - tick_len], color="black", linewidth=0.6)
        ax.text(x, y - 2.2 * tick_len, f"{val}", ha="center", va="top", fontsize=font_size)

        # MgO ticks along left edge (FeO=0)
        x, y = ternary_to_xy(0.0, 100.0 - val, val)
        ax.plot([x - tick_len, x], [y + tick_len * 0.6, y], color="black", linewidth=0.6)
        ax.text(x - 2.2 * tick_len, y + tick_len * 0.6, f"{val}", ha="right", va="center", fontsize=font_size)

        # SiO2 ticks along right edge (MgO=0)
        x, y = ternary_to_xy(100.0 - val, val, 0.0)
        ax.plot([x, x + tick_len], [y, y + tick_len * 0.6], color="black", linewidth=0.6)
        ax.text(x + 2.2 * tick_len, y + tick_len * 0.6, f"{val}", ha="left", va="center", fontsize=font_size)


def draw_triangle(ax):
    h = math.sqrt(3.0) / 2.0
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h], [0.0, 0.0]])
    ax.plot(triangle[:, 0], triangle[:, 1], color="black", linewidth=1.0)

    # Axis labels
    ax.text(-0.02, -0.03, "MgO", ha="left", va="top", fontsize=10)
    ax.text(1.02, -0.03, "FeO", ha="right", va="top", fontsize=10)
    ax.text(0.5, h + 0.03, "SiO2", ha="center", va="bottom", fontsize=10)

    # Corner values
    ax.text(-0.01, 0.02, "100", ha="right", va="bottom", fontsize=8)
    ax.text(1.01, 0.02, "100", ha="left", va="bottom", fontsize=8)
    ax.text(0.5, h + 0.015, "100", ha="center", va="bottom", fontsize=8)

    draw_ternary_grid(ax, step=10)
    draw_ternary_ticks(ax, step=10)

    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, h + 0.08)
    ax.axis("off")


def plot_exp_on_ternary(ax, filename, color):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    try:
        with open(file_path, "r", encoding="latin-1") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"No se encontro {filename}")
        return

    blocks, metadata = parse_thermocalc_exp(content)
    scale = detect_scale(blocks, metadata)

    cao_pct = extract_cao_from_filename(filename)
    total_tri = 100.0 - AL2O3_PCT - MNO_PCT - cao_pct
    allowed_liquids = {"IONIC_LIQ#1", "IONIC_LIQ#2", "IONIC_LIQ#3"}
    allowed_phases = allowed_liquids | {"HALITE#1", "HALITE#2"}

    segments_xy = []
    for block in blocks:
        phases_set = set(block["phases"])
        has_halite = "HALITE#1" in phases_set or "HALITE#2" in phases_set
        only_liquid_halite = phases_set.issubset(allowed_phases)
        if not (has_halite and only_liquid_halite):
            continue
        block_id = block.get("block_id")
        if filename == "40.exp" and block_id in EXCLUDED_BLOCK_IDS_40:
            continue
        phase_label = format_phase_label(block["phases"])
        if filename == "35.exp" and phase_label in EXCLUDED_PHASE_LABELS_BLUE:
            continue

        for segment in block["segments"]:
            if len(segment) == 0:
                continue

            seg_scaled = segment * scale
            valid_points = []

            for feo, sio2 in seg_scaled:
                mgo = total_tri - feo - sio2
                if mgo < 0:
                    if len(valid_points) >= 2:
                        xy = np.array([ternary_to_xy(*p) for p in valid_points])
                        segments_xy.append((xy, phase_label))
                    valid_points = []
                    continue
                valid_points.append((feo, sio2, mgo))

            if len(valid_points) >= 2:
                xy = np.array([ternary_to_xy(*p) for p in valid_points])
                segments_xy.append((xy, phase_label))

    if not segments_xy:
        return

    total = len(segments_xy)
    for idx, (xy, phase_label) in enumerate(segments_xy):
        t = idx / max(total - 1, 1)
        seg_color = warm_shade(color, t)
        ax.plot(xy[:, 0], xy[:, 1], color=seg_color, linewidth=0.8)

def plot_eaf_points(ax):
    points = puntos_EAF_completos()
    xs = []
    ys = []
    for p in points:
        xy = ternary_to_xy(p["FeO"], p["SiO2"], p["MgO"])
        if xy is None:
            continue
        xs.append(xy[0])
        ys.append(xy[1])

    ax.scatter(xs, ys, c="red", edgecolors="black", linewidth=0.4, s=18, zorder=6)


# ==========================================
# MAIN
# ==========================================

def run_graf3():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    width = DOUBLE_COLUMN_MM * MM_TO_INCH
    height = width
    fig, ax = plt.subplots(figsize=(width, height))

    draw_triangle(ax)

    for filename in FILES_TO_PROCESS:
        color = FILE_COLORS.get(filename, "black")
        plot_exp_on_ternary(ax, filename, color)

    plot_eaf_points(ax)

    plt.tight_layout()
    output_path = os.path.join(figures_dir, "ternario_30_35_40.png")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Guardado: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    run_graf3()
