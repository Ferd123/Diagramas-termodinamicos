#%%
import math
import os
import re
from typing import Any, List, Optional, TypedDict, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpltern  # noqa: F401
import numpy as np

# ==========================================
# CONSTANTS & CONFIG
# ==========================================
SINGLE_COLUMN_MM = 85
DOUBLE_COLUMN_MM = 170
MM_TO_INCH = 1 / 25.4

#%%
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
    "PCTFEO": "% FeO",
    "PCTSIO2": "% SiO$_2$",
    "PCTMGO": "% MgO",
    "PCTCAO": "% CaO",
}

AL2O3_PCT = 8.0
MNO_PCT = 4.0

# Offsets para mover los numeros de zonas (dx, dy) en coordenadas del eje (0-1).
# Ejemplo: ZONE_LABEL_OFFSETS = {35.0: {2: (0.02, -0.01)}}
ZONE_LABEL_OFFSETS = {
    35.0: {
        1: (0.1, 0.1),
        2: (0.15, -0.1),
        3: (-0.1, -0.05),
        5: (-0.05, -0.01),
        4: (0.2,-0.25),
        6: (-0.05,0),
        7: (-0.01, 0.015),
        8:(0.05,-0.05)
    },
    40.0: {
        1: (0.2, 0.2),
        2: (0.12, -0.1),
        3: (-0.1, -0.05),
        5: (-0.05, -0.01),
        4: (0.16,-0.25),
        6: (-0.05,0),
        7: (-0.01, 0.015),
        9:(-0.2,0.2)
    },
}
# Posicion del cuadro de fases (x, y) en coordenadas del eje (0-1).
# Ejemplo: ZONE_LEGEND_POS = {35.0: (0.05, 0.95), @40.0: (0.1, 0.9)}
ZONE_LEGEND_POS = {
    35.0: (-0.2, 0.95),
    40.0: (-0.2, 0.9),
}
# Posicion del texto "LIQUID" (x, y) en coordenadas del eje (0-1).
# Ejemplo: LIQUID_TEXT_POS = {35.0: (0.55, 0.45), 40.0: (0.6, 0.5)}
LIQUID_TEXT_POS = {
    30: (0.57,0.6),
    40.0: (0.57, 0.6),
    35.0: (0.55, 0.6),
}
LIQUID_TEXT_DEFAULT = (0.5, 0.5)


#%%
class ThermoCalcMetadata(TypedDict):
    xscale: Optional[List[float]]
    yscale: Optional[List[float]]
    xtext: Optional[str]
    ytext: Optional[str]


#%%
def parse_thermocalc_exp(content):
    blocks = []
    current_block = None
    metadata: ThermoCalcMetadata = {
        "xscale": None,
        "yscale": None,
        "xtext": None,
        "ytext": None,
    }

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


#%%
def parse_cao_from_filename(base_name):
    matches = re.findall(r"(\d+(?:[.,]\d+)?)", base_name)
    if not matches:
        return None
    value = matches[0].replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return None


#%%
def get_base_dir():
    if "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


def plot_phase_diagram(
    exp_path,
    figures_dir=None,
    *,
    al2o3_pct=AL2O3_PCT,
    mno_pct=MNO_PCT,
    show=True,
    save=False,
):
    with open(exp_path, "r", encoding="latin-1") as f:
        content = f.read()

    blocks, metadata = parse_thermocalc_exp(content)
    if not blocks:
        return None

    scale = detect_scale(blocks)
    x_label = extract_label(metadata.get("xtext")) or "FeO (%)"
    y_label = extract_label(metadata.get("ytext")) or "SiO$_2$ (%)"

    base_name = os.path.basename(exp_path)
    cao_pct = parse_cao_from_filename(base_name)

    if cao_pct is None:
        print(f"No se pudo extraer CaO desde {base_name}, se omite.")
        return None

    total_tri = 100.0 - al2o3_pct - mno_pct - cao_pct
    if total_tri <= 0:
        print(f"Total ternario invalido para {base_name}, se omite.")
        return None

    width = DOUBLE_COLUMN_MM * MM_TO_INCH
    height = width
    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor("white")
    tax = cast(Any, fig.add_subplot(111, projection="ternary"))
    tax.set_tlim(0, total_tri)
    tax.set_llim(0, total_tri)
    tax.set_rlim(0, total_tri)
    tax.grid(True, linewidth=0.6, color="#c7c7c7", linestyle="--")

    tick_vals = list(np.arange(10, math.floor(total_tri / 10) * 10 + 0.1, 10))
    tax.taxis.set_ticks([])
    tax.laxis.set_ticks([])
    tax.raxis.set_ticks([])

    def ternary_to_axes_fraction(feo, sio2, mgo):
        total = feo + sio2 + mgo
        if total <= 0:
            return None
        feo /= total
        sio2 /= total
        x = feo + 0.5 * sio2
        y = (math.sqrt(3.0) / 2.0) * sio2
        return x, y / (math.sqrt(3.0) / 2.0)

    tick_len = 0.02
    label_offset = 0.04
    left_normal = (-0.894, 0.447)
    right_normal = (0.894, 0.447)
    for v in tick_vals:
        # FeO ticks along base (SiO2=0)
        pos = ternary_to_axes_fraction(v, 0.0, total_tri - v)
        if pos:
            x, y = (float(pos[0]), float(pos[1]))
            tax.plot([x, x], [y, y - tick_len], transform=tax.transAxes,
                    color="black", linewidth=0.6, clip_on=False)
            tax.text(x, y - label_offset, f"{v:g}", transform=tax.transAxes,
                    ha="center", va="top", fontsize=9, clip_on=False)

        # MgO ticks along left edge (FeO=0)
        pos = ternary_to_axes_fraction(0.0, total_tri - v, v)
        if pos:
            x, y = (float(pos[0]), float(pos[1]))
            tax.plot(
                [x, x + left_normal[0] * tick_len],
                [y, y + left_normal[1] * tick_len],
                transform=tax.transAxes,
                color="black",
                linewidth=0.6,
                clip_on=False,
            )
            tax.text(
                x + left_normal[0] * label_offset,
                y + left_normal[1] * label_offset,
                f"{v:g}",
                transform=tax.transAxes,
                ha="right",
                va="center",
                fontsize=9,
                clip_on=False,
            )

        # SiO2 ticks along right edge (MgO=0)
        pos = ternary_to_axes_fraction(total_tri - v, v, 0.0)
        if pos:
            x, y = (float(pos[0]), float(pos[1]))
            tax.plot(
                [x, x + right_normal[0] * tick_len],
                [y, y + right_normal[1] * tick_len],
                transform=tax.transAxes,
                color="black",
                linewidth=0.6,
                clip_on=False,
            )
            tax.text(
                x + right_normal[0] * label_offset,
                y + right_normal[1] * label_offset,
                f"{v:g}",
                transform=tax.transAxes,
                ha="left",
                va="center",
                fontsize=9,
                clip_on=False,
            )

    # Ejes solicitados: izquierda MgO, derecha FeO, arriba SiO2
    max_label = f"{total_tri:.1f}%"
    left_label = f"{max_label} MgO"
    right_label = f"{max_label} FeO"
    top_label = f"{max_label} SiO$_2$"
    tax.set_llabel("")
    tax.set_rlabel("")
    tax.set_tlabel(top_label, fontsize=10, labelpad=14)
    tax.text(
        -0.15,
        -0.02,
        left_label,
        transform=tax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        rotation=0,
        clip_on=False,
    )
    tax.text(
        1.15,
        -0.02,
        right_label,
        transform=tax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        rotation=0,
        clip_on=False,
    )

    colors = [
        "#0b1f3a", "#2a6f97", "#52b788", "#ef476f", "#f4a261",
        "#8338ec", "#3d405b", "#8d99ae", "#264653",
    ]

    zone_map = {}
    zone_order = []
    zone_centers = {}
    zone_points = {}
    zone_label_parts = {}
    has_liquid = False
    has_liquid_mgo = False
    grouped_ids = {10: 7, 7: 7, 9: 8, 8: 8}

    def normalize_phases(phases):
        clean = []
        for name in phases:
            name = name.replace("#", "")
            if name.startswith("IONIC_LIQ"):
                name = "Liquid"
            elif "ca2sio4_alpha_a" in name.lower():
                name = "Ca₂SiO₄"
            elif "hatrurite" in name.lower():
                name = "3CaO.SiO₂"
            elif name.startswith("HALITE1"):
                name = "MgO"
            elif name.startswith("HALITE2"):
                name = "(Fe,Mg)O"
            if name not in clean:
                clean.append(name)
        return clean

    def format_phases(clean):
        return " + ".join(clean) if clean else "Unknown"

    def place_zone_label(zone_id, c_t, c_l, c_r):
        pos = ternary_to_axes_fraction(c_r, c_t, c_l)
        if not pos:
            return
        x, y = (float(pos[0]), float(pos[1]))
        dx, dy = (0.0, 0.0)
        if cao_pct in ZONE_LABEL_OFFSETS:
            dx, dy = ZONE_LABEL_OFFSETS.get(cao_pct, {}).get(zone_id, (0.0, 0.0))
        tax.text(
            x + dx,
            y + dy,
            str(zone_id),
            transform=tax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="black",
        )

    for block in blocks:
        phases_set = set(block["phases"])
        color_seed = sum(map(ord, "".join(sorted(phases_set))))
        color = colors[color_seed % len(colors)]
        clean_phases = normalize_phases(block["phases"])
        zone_key = format_phases(clean_phases)
        if "Liquid" in clean_phases:
            has_liquid = True
        if "Liquid" in clean_phases and "MgO" in clean_phases and len(clean_phases) == 2:
            has_liquid_mgo = True
        if cao_pct != 30.0:
            if zone_key not in zone_map:
                zone_map[zone_key] = len(zone_map) + 1
                zone_order.append(zone_key)
            zone_id = zone_map[zone_key]
            parts = zone_label_parts.setdefault(zone_id, set())
            for part in clean_phases:
                parts.add(part)

        block_points = []
        for segment in block["segments"]:
            if len(segment) == 0:
                continue
            seg_scaled = segment * scale
            t_vals = []
            l_vals = []
            r_vals = []
            centers = []
            for feo, sio2 in seg_scaled:
                mgo = total_tri - feo - sio2
                if mgo < 0:
                    if len(t_vals) >= 2:
                        tax.plot(np.asarray(t_vals), np.asarray(l_vals), np.asarray(r_vals),
                                color=color, linewidth=0.8)
                    t_vals = []
                    l_vals = []
                    r_vals = []
                    centers = []
                    continue
                t_vals.append(sio2)
                l_vals.append(mgo)
                r_vals.append(feo)
                centers.append((sio2, mgo, feo))
                block_points.append((sio2, mgo, feo))
                if cao_pct != 30.0:
                    zone_id = grouped_ids.get(zone_id, zone_id)
                    zone_points.setdefault(zone_id, []).append((sio2, mgo, feo))
            if len(t_vals) >= 2:
                tax.plot(np.asarray(t_vals), np.asarray(l_vals), np.asarray(r_vals),
                        color=color, linewidth=1.1)
                if cao_pct != 30.0 and centers:
                    c_t = float(np.mean([c[0] for c in centers]))
                    c_l = float(np.mean([c[1] for c in centers]))
                    c_r = float(np.mean([c[2] for c in centers]))
                    if cao_pct in (35.0, 40.0):
                        zone_id = grouped_ids.get(zone_id, zone_id)
                        zone_centers.setdefault(zone_id, []).append((c_t, c_l, c_r))
                    else:
                        zone_id = grouped_ids.get(zone_id, zone_id)
                        place_zone_label(zone_id, c_t, c_l, c_r)
        if cao_pct in (35.0, 40.0) and block_points:
            c_t = float(np.mean([p[0] for p in block_points]))
            c_l = float(np.mean([p[1] for p in block_points]))
            c_r = float(np.mean([p[2] for p in block_points]))
            zone_id = grouped_ids.get(zone_id, zone_id)
            zone_centers.setdefault(zone_id, []).append((c_t, c_l, c_r))

    tax.set_title(
        f"Diagrama ternario (Al$_2$O$_3$ {al2o3_pct}%, MnO {mno_pct}%) - CaO{base_name.replace('.exp', '')}%",
        fontsize=11,
        pad=14,
    )
    if cao_pct in (35.0, 40.0):
        # Etiquetar todas las zonas detectadas usando los puntos del bloque.
        for zone_id, pts in zone_points.items():
            if not pts:
                continue
            c_t = float(np.mean([p[0] for p in pts]))
            c_l = float(np.mean([p[1] for p in pts]))
            c_r = float(np.mean([p[2] for p in pts]))
            place_zone_label(zone_id, c_t, c_l, c_r)
        # Si aun falta alguna zona, usar el punto con mayor FeO como respaldo.
        for zone_key in zone_order:
            zone_id = grouped_ids.get(zone_map[zone_key], zone_map[zone_key])
            if zone_id in zone_points:
                continue
            pts = zone_points.get(zone_id, [])
            if not pts:
                continue
            max_point = max(pts, key=lambda p: p[2])
            place_zone_label(zone_id, float(max_point[0]), float(max_point[1]), float(max_point[2]))
    if has_liquid:
        pos = LIQUID_TEXT_POS.get(cao_pct, LIQUID_TEXT_DEFAULT)
        tax.text(
            pos[0],
            pos[1],
            "LIQUID",
            transform=tax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            clip_on=False,
        )
    if cao_pct == 30.0:
        if has_liquid_mgo:
            tax.text(
                0.5,
                0.2,
                "Liquid + MgO",
                transform=tax.transAxes,
                ha="center",
                va="top",
                fontsize=12,
                clip_on=False,
            )
    elif zone_order:
        legend_pos = ZONE_LEGEND_POS.get(cao_pct, (0.02, 0.98))
        grouped_labels = {}
        for zone_key in zone_order:
            zone_id = zone_map[zone_key]
            group_id = grouped_ids.get(zone_id, zone_id)
            parts = zone_label_parts.get(zone_id, set())
            grouped_labels.setdefault(group_id, set()).update(parts)
        legend_lines = []
        used_groups = set()
        for zone_key in zone_order:
            zone_id = zone_map[zone_key]
            group_id = grouped_ids.get(zone_id, zone_id)
            if group_id in used_groups:
                continue
            used_groups.add(group_id)
            parts = sorted(grouped_labels.get(group_id, set()))
            label = " + ".join(parts) if parts else zone_key
            legend_lines.append(f"{group_id} = {label}")
        if cao_pct == 35.0:
            legend_lines = [
                line for line in legend_lines
                if line.split(" = ", 1)[-1].strip() != "Liquid"
            ]
            renumbered = []
            for idx, line in enumerate(legend_lines, start=1):
                _, label = line.split(" = ", 1)
                renumbered.append(f"{idx} = {label}")
            legend_lines = renumbered
        tax.text(
            legend_pos[0],
            legend_pos[1],
            "\n".join(legend_lines),
            transform=tax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "none", "edgecolor": "none"},
        )
    plt.tight_layout()
    output_path = None
    if save:
        if figures_dir is None:
            raise ValueError("figures_dir es requerido cuando save=True.")
        output_path = os.path.join(
            figures_dir,
            f"diagrama_fases_{os.path.basename(exp_path).replace('.exp', '')}.png",
        )
        plt.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


#%%
def run_diagramas():
    script_dir = get_base_dir()
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


#%%
def interactive_diagramas(base_dir=None):
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    base_dir = base_dir or get_base_dir()
    exp_files = sorted(
        f for f in os.listdir(base_dir) if f.lower().endswith(".exp")
    )
    if not exp_files:
        print("No se encontraron archivos .exp en la carpeta seleccionada.")
        return

    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    file_widget = widgets.Dropdown(options=exp_files, description="Archivo")
    al2o3_widget = widgets.FloatSlider(
        value=AL2O3_PCT, min=0.0, max=20.0, step=0.5, description="Al2O3 (%)"
    )
    mno_widget = widgets.FloatSlider(
        value=MNO_PCT, min=0.0, max=20.0, step=0.5, description="MnO (%)"
    )
    save_widget = widgets.Checkbox(value=False, description="Guardar PNG")
    output = widgets.Output()

    def render(file_name, al2o3_pct, mno_pct, save_png):
        exp_path = os.path.join(base_dir, file_name)
        with output:
            clear_output(wait=True)
            plot_phase_diagram(
                exp_path,
                figures_dir=figures_dir,
                al2o3_pct=al2o3_pct,
                mno_pct=mno_pct,
                show=True,
                save=save_png,
            )

    controls = widgets.VBox([file_widget, al2o3_widget, mno_widget, save_widget])
    ui = widgets.HBox([controls, output])
    out = widgets.interactive_output(
        render,
        {
            "file_name": file_widget,
            "al2o3_pct": al2o3_widget,
            "mno_pct": mno_widget,
            "save_png": save_widget,
        },
    )
    display(ui, out)


#%%
if __name__ == "__main__":
    run_diagramas()
