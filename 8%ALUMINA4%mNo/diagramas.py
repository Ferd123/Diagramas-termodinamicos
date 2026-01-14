#%%
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpltern  # noqa: F401
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================

SINGLE_COLUMN_MM = 85
DOUBLE_COLUMN_MM = 170
MM_TO_INCH = 1 / 25.4

AL2O3_PCT_DEFAULT = 8.0
MNO_PCT_DEFAULT = 4.0

LABEL_MAP = {
    "PCTFEO": "% FeO",
    "PCTSIO2": "% SiO$_2$",
    "PCTMGO": "% MgO",
    "PCTCAO": "% CaO",
}

# Offsets para mover los numeros de zonas (dx, dy) en coords del eje (0-1)
ZONE_LABEL_OFFSETS: Dict[float, Dict[int, Tuple[float, float]]] = {
    30.0: {
        1: (0.1, 0.1),
        2: (0.15, -0.1),
        3: (-0.1, -0.05),
        4: (0.2, -0.25),
        5: (-0.05, -0.01),
        6: (-0.05, 0),
        7: (-0.01, 0.015),
        8: (0.05, -0.05),
        9: (0.2, 0)
    },
    35.0: {
        2: (0.35, -0.4),
        3: (-0.1, -0.05),
        5: (-0.05, -0.01),
        6: (0.05, -0.1),  # Adjusted for arrow
        7: (-0.15, 0), # Adjusted for arrow
        8: (0.05, -0.05),
        9: (0.05, 0.05),
        10: (0.1, 0.1),   # Added for arrow
    },
    40.0: {
        1: (0.1, 0.05), # Arrow needed
        2: (0.12, -0.1),
        3: (-0.1, -0.05),
        4: (-0.12, 0.05),
        5: (-0.05, -0.01),
        6: (-0.15, 0.1), # Arrow needed
        7: (-0.15, 0.05), # Adjusted for arrow
        9: (0.4, -0.47),
    },
}

# Posición del texto "LIQUID"
LIQUID_TEXT_POS = {
    30.0: (0.57, 0.6),
    35.0: (0.55, 0.6),
    40.0: (0.57, 0.6),
}
LIQUID_TEXT_DEFAULT = (0.5, 0.5)

# Cuadro de leyenda por CaO
ZONE_LEGEND_POS = {
    35.0: (-0.2, 1.1),
    40.0: (-0.2, 1.1),
}

# Paleta (puedes cambiarla si quieres)
COLORS = [
    "#0b1f3a", "#2a6f97", "#52b788", "#ef476f", "#f4a261",
    "#8338ec", "#3d405b", "#8d99ae", "#264653", "#e76f51", "#f4a261",
    "#e9c46a", "#2a9d8f", "#264653",
]

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


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ThermoCalcMetadata:
    xscale: Optional[List[float]] = None
    yscale: Optional[List[float]] = None
    xtext: Optional[str] = None
    ytext: Optional[str] = None


@dataclass
class Block:
    phases_raw: List[str]
    segments: List[np.ndarray]  # list of Nx2 arrays


# =============================================================================
# PARSING
# =============================================================================

_NUMBER_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


def parse_thermocalc_exp(content: str) -> Tuple[List[Block], ThermoCalcMetadata]:
    """
    Parsea .exp de Thermo-Calc en bloques:
      - fases asociadas a $F0 / $E
      - segmentos (polilíneas) como arrays Nx2
    """
    blocks: List[Block] = []
    metadata = ThermoCalcMetadata()

    header_starts = (
        "PROLOG", "XSCALE", "YSCALE", "XTYPE", "YTYPE", "XLENGTH", "YLENGTH",
        "TITLE", "XTEXT", "YTEXT", "DATASET", "CHAR", "COLOR",
        "$ BLOCK", "$BLOCK", "$F0", "$E", "BLOCK", "BLOCKEND"
    )

    current_phases: List[str] = []
    current_segments: List[np.ndarray] = []
    current_segment: List[List[float]] = []

    def flush_segment():
        nonlocal current_segment, current_segments
        if current_segment:
            current_segments.append(np.array(current_segment, dtype=float))
            current_segment = []

    def flush_block():
        nonlocal current_phases, current_segments
        if current_phases or current_segments:
            blocks.append(Block(phases_raw=current_phases[:], segments=current_segments[:]))
        current_phases = []
        current_segments = []

    for raw_line in content.split("\n"):
        line = raw_line.strip()
        if not line:
            flush_segment()
            continue

        # metadata
        if line.startswith("XSCALE"):
            nums = [float(x) for x in _NUMBER_RE.findall(line)]
            if len(nums) >= 2:
                metadata.xscale = nums
        elif line.startswith("YSCALE"):
            nums = [float(x) for x in _NUMBER_RE.findall(line)]
            if len(nums) >= 2:
                metadata.yscale = nums
        elif line.startswith("XTEXT"):
            metadata.xtext = line
        elif line.startswith("YTEXT"):
            metadata.ytext = line

        # blocks
        if line.startswith("$ BLOCK") or line.startswith("$BLOCK"):
            flush_block()
            continue

        if line.startswith("$F0") or line.startswith("$E"):
            parts = line.split(maxsplit=1)
            phase_name = parts[1] if len(parts) > 1 else "Unknown"
            current_phases.append(phase_name)
            continue

        if line.startswith("BLOCKEND"):
            flush_segment()
            flush_block()
            continue

        if line.startswith("BLOCK") or line.startswith("$"):
            continue

        # ignore headers inside a block
        if line.startswith(header_starts):
            flush_segment()
            continue

        # coordinates
        nums = [float(x) for x in _NUMBER_RE.findall(line)]
        if len(nums) >= 2:
            x_val, y_val = nums[0], nums[1]
            # Thermo-Calc marker "M" = new segment
            if "M" in line:
                flush_segment()
                current_segment.append([x_val, y_val])
            else:
                current_segment.append([x_val, y_val])
        else:
            flush_segment()

    flush_segment()
    flush_block()
    return blocks, metadata


def detect_scale(blocks: List[Block]) -> float:
    """
    Si los datos vienen en 0-1, escala a 100.
    Si vienen en 0-100, deja en 1.
    """
    max_val = 0.0
    for b in blocks:
        for seg in b.segments:
            if seg.size == 0:
                continue
            max_val = max(max_val, float(np.max(seg)))
    return 100.0 if max_val <= 1.0 else 1.0


def extract_label(text_line: Optional[str]) -> Optional[str]:
    if not text_line:
        return None
    match = re.search(r"(PCT[A-Z0-9]+|W\([A-Z0-9]+\))", text_line)
    if not match:
        return None
    key = match.group(1)
    if key.startswith("W(") and key.endswith(")"):
        key = f"PCT{key[2:-1]}"
    return LABEL_MAP.get(key, key)


def parse_cao_from_filename(base_name: str) -> Optional[float]:
    matches = re.findall(r"(\d+(?:[.,]\d+)?)", base_name)
    if not matches:
        return None
    value = matches[0].replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return None


def get_base_dir() -> str:
    if "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


# =============================================================================
# PHASE NORMALIZATION (NO GROUPING EXCEPT LIQUIDS)
# =============================================================================

_LIQ_RE = re.compile(r"(?:IONIC[_\s-]?LIQ|LIQUID)(?:[_\s-]?(\d+))?", re.IGNORECASE)

def _clean_phase_token(s: str) -> str:
    return s.replace("#", "").strip()

def normalize_phase_name(name: str) -> str:
    """
    Normaliza nombres individuales SIN agrupar fases.
    Sólo corrige alias obvios y estandariza.
    """
    n = _clean_phase_token(name)

    nl = n.lower()

    # Ejemplos de alias / pretty names (ajusta aquí sin tocar el resto del plot)
    if "ca2sio4_alpha_a" in nl:
        return "Ca₂SiO₄"
    if "hatrurite" in nl:
        return "3CaO·SiO₂"
    if n.startswith("HALITE1"):
        return "MgO"
    if n.startswith("HALITE2"):
        return "(Fe,Mg)O"

    # Líquidos se manejan en una segunda pasada (assign_liquid_variants)
    m = _LIQ_RE.search(n)
    if m:
        # Deja un placeholder "Liquid?" con índice si existe
        idx = m.group(1)
        if idx:
            return f"Liquid {idx}"
        return "Liquid"  # sin índice aún

    return n


def assign_liquid_variants(phases: List[str], cao_pct: Optional[float] = None) -> List[str]:
    """
    Regla pedida:
      - No agrupar nada excepto líquidos.
      - Si existen Liquid 1 y Liquid 2 (o 2 y 3, etc) conservarlos.
      - Si no se puede inferir índice, colapsar a Liquid 1 si solo hay un líquido.
      - Excepción: Si cao_pct == 35.0, forzar TODO "Liquid X" a "Liquid".
    """
    normalized = [normalize_phase_name(p) for p in phases]

    # Regla especial para 35%: todo líquido es "Liquid" a secas
    if cao_pct == 35.0:
        return ["Liquid" if p.startswith("Liquid") else p for p in normalized]

    # detectar líquidos
    liquid_indices = set()
    liquid_positions: List[int] = []
    for i, p in enumerate(normalized):
        if p.startswith("Liquid"):
            liquid_positions.append(i)
            m = re.match(r"Liquid\s+(\d+)$", p)
            if m:
                liquid_indices.add(int(m.group(1)))

    if not liquid_positions:
        return normalized

    # Caso: líquidos sin índice -> decidir si hay 1 o varios
    # Si ya hay índices, respetarlos; si hay "Liquid" sin número y hay varios índices, asigna al más bajo disponible.
    if liquid_indices:
        # completar los "Liquid" sin índice:
        for pos in liquid_positions:
            if normalized[pos] == "Liquid":
                # asigna el menor índice existente (conservador)
                normalized[pos] = f"Liquid {min(liquid_indices)}"
        return normalized

    # No había índices explícitos (todo era "Liquid" sin número):
    # Si hay más de un líquido en el bloque, no podemos separarlos sin info -> por tu regla:
    # los dejamos como Liquid 1 y Liquid 2 si hay 2 ocurrencias, Liquid 1/2/3 si 3, etc.
    # Esto es "separar" por ocurrencia, no por Thermo-Calc real, pero cumple lo que pediste en ausencia de índice.
    count = len(liquid_positions)
    for k, pos in enumerate(liquid_positions, start=1):
        normalized[pos] = f"Liquid {k}"
    return normalized


def format_zone_key(phases: List[str]) -> str:
    """
    Clave canónica para identificar zona (conjunto de fases), sin agrupar.
    """
    uniq = []
    for p in phases:
        if p not in uniq:
            uniq.append(p)
    # orden estable para evitar cambios
    return " + ".join(sorted(uniq))


# =============================================================================
# PLOTTING
# =============================================================================

def ternary_to_axes_fraction(feo: float, sio2: float, mgo: float) -> Optional[Tuple[float, float]]:
    total = feo + sio2 + mgo
    if total <= 0:
        return None
    feo /= total
    sio2 /= total
    x = feo + 0.5 * sio2
    y = (math.sqrt(3.0) / 2.0) * sio2
    return x, y / (math.sqrt(3.0) / 2.0)


def draw_custom_ticks(tax: Any, total_tri: float) -> None:
    tick_vals = list(np.arange(10, math.floor(total_tri / 10) * 10 + 0.1, 10))
    tax.taxis.set_ticks([])
    tax.laxis.set_ticks([])
    tax.raxis.set_ticks([])

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
            tax.plot([x, x + left_normal[0] * tick_len],
                     [y, y + left_normal[1] * tick_len],
                     transform=tax.transAxes, color="black",
                     linewidth=0.6, clip_on=False)
            tax.text(x + left_normal[0] * label_offset,
                     y + left_normal[1] * label_offset,
                     f"{v:g}", transform=tax.transAxes,
                     ha="right", va="center", fontsize=9, clip_on=False)

        # SiO2 ticks along right edge (MgO=0)
        pos = ternary_to_axes_fraction(total_tri - v, v, 0.0)
        if pos:
            x, y = (float(pos[0]), float(pos[1]))
            tax.plot([x, x + right_normal[0] * tick_len],
                     [y, y + right_normal[1] * tick_len],
                     transform=tax.transAxes, color="black",
                     linewidth=0.6, clip_on=False)
            tax.text(x + right_normal[0] * label_offset,
                     y + right_normal[1] * label_offset,
                     f"{v:g}", transform=tax.transAxes,
                     ha="left", va="center", fontsize=9, clip_on=False)


def set_axis_titles(tax: Any, total_tri: float) -> None:
    # Ejes solicitados: izquierda MgO, derecha FeO, arriba SiO2
    max_label = f"{total_tri:.1f}%"
    left_label = f"{max_label} MgO"
    right_label = f"{max_label} FeO"
    top_label = f"{max_label} SiO$_2$"

    tax.set_llabel("")
    tax.set_rlabel("")
    tax.set_tlabel(top_label, fontsize=10, labelpad=14)

    tax.text(-0.15, -0.02, left_label, transform=tax.transAxes,
             ha="left", va="top", fontsize=10, rotation=0, clip_on=False)
    tax.text(1.15, -0.02, right_label, transform=tax.transAxes,
             ha="right", va="top", fontsize=10, rotation=0, clip_on=False)


def plot_phase_diagram(
    exp_path: str,
    figures_dir: Optional[str] = None,
    *,
    al2o3_pct: float = AL2O3_PCT_DEFAULT,
    mno_pct: float = MNO_PCT_DEFAULT,
    show: bool = True,
    save: bool = False,
) -> Optional[str]:
    with open(exp_path, "r", encoding="latin-1") as f:
        content = f.read()

    blocks, metadata = parse_thermocalc_exp(content)
    if not blocks:
        return None

    scale = detect_scale(blocks)
    _ = extract_label(metadata.xtext) or "FeO (%)"
    _ = extract_label(metadata.ytext) or "SiO$_2$ (%)"

    base_name = os.path.basename(exp_path)
    cao_pct = parse_cao_from_filename(base_name)
    if cao_pct is None:
        print(f"No se pudo extraer CaO desde {base_name}, se omite.")
        return None

    total_tri = 100.0 - al2o3_pct - mno_pct - cao_pct
    if total_tri <= 0:
        print(f"Total ternario inválido para {base_name}, se omite.")
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

    draw_custom_ticks(tax, total_tri)
    set_axis_titles(tax, total_tri)

    # zone registry: zone_key -> id
    zone_map: Dict[str, int] = {}
    zone_order: List[str] = []
    zone_points: Dict[int, List[Tuple[float, float, float]]] = {}
    zone_label_parts: Dict[int, List[str]] = {}

    has_any_liquid = False

    def get_color_for_zone(zone_key: str) -> str:
        seed = sum(map(ord, zone_key))
        return COLORS[seed % len(COLORS)]

    def place_zone_label(zone_id: int, c_t: float, c_l: float, c_r: float) -> None:
        pos = ternary_to_axes_fraction(c_r, c_t, c_l)
        if not pos:
            return
        x, y = float(pos[0]), float(pos[1])
        dx, dy = (0.0, 0.0)
        if cao_pct in ZONE_LABEL_OFFSETS:
            dx, dy = ZONE_LABEL_OFFSETS[cao_pct].get(zone_id, (0.0, 0.0))
        
        # Check if this requires an arrow 
        # 35% CaO: zones 6, 7, 10
        # 40% CaO: zones 1, 7
        need_arrow = False
        if cao_pct == 35.0 and zone_id in [6, 7, 10]:
            need_arrow = True
        elif cao_pct == 40.0 and zone_id in [1, 6, 7]:
            need_arrow = True

        if need_arrow:
             tax.annotate(
                str(zone_id),
                xy=(x, y), xycoords='axes fraction',
                xytext=(x + dx, y + dy), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="-", color="black", linewidth=0.8),
                ha="center", va="center", fontsize=9, color="black"
            )
        else:
            tax.text(x + dx, y + dy, str(zone_id), transform=tax.transAxes,
                     ha="center", va="center", fontsize=9, color="black")

    for b in blocks:
        # 1) normaliza fases sin agrupar, 2) asigna variantes de liquid si aplica
        clean_phases = assign_liquid_variants(b.phases_raw, cao_pct=cao_pct)

        if any(p.startswith("Liquid") for p in clean_phases):
            has_any_liquid = True

        zone_key = format_zone_key(clean_phases)

        if zone_key not in zone_map:
            zone_map[zone_key] = len(zone_map) + 1
            zone_order.append(zone_key)
            zone_label_parts[zone_map[zone_key]] = sorted(set(clean_phases))

        zone_id = zone_map[zone_key]
        color = get_color_for_zone(zone_key)

        for seg in b.segments:
            if seg.size == 0:
                continue
            seg_scaled = seg * scale

            t_vals: List[float] = []
            l_vals: List[float] = []
            r_vals: List[float] = []
            pts_for_zone: List[Tuple[float, float, float]] = []

            for feo, sio2 in seg_scaled:
                mgo = total_tri - feo - sio2
                if mgo < 0:
                    if len(t_vals) >= 2:
                        tax.plot(np.asarray(t_vals), np.asarray(l_vals), np.asarray(r_vals),
                                 color=color, linewidth=1.1)
                    t_vals, l_vals, r_vals, pts_for_zone = [], [], [], []
                    continue

                t_vals.append(float(sio2))
                l_vals.append(float(mgo))
                r_vals.append(float(feo))
                pts_for_zone.append((float(sio2), float(mgo), float(feo)))

            if len(t_vals) >= 2:
                tax.plot(np.asarray(t_vals), np.asarray(l_vals), np.asarray(r_vals),
                         color=color, linewidth=1.1)
                if pts_for_zone:
                    zone_points.setdefault(zone_id, []).extend(pts_for_zone)

    # Etiquetas (centroide) por zona
    if cao_pct != 30.0:
        for zid, pts in zone_points.items():
            if not pts:
                continue
            c_t = float(np.mean([p[0] for p in pts]))
            c_l = float(np.mean([p[1] for p in pts]))
            c_r = float(np.mean([p[2] for p in pts]))
            place_zone_label(zid, c_t, c_l, c_r)

    tax.set_title(
        f"Diagrama ternario (Al$_2$O$_3$ {al2o3_pct}%, MnO {mno_pct}%) - CaO {cao_pct:g}%",
        fontsize=11,
        pad=14,
    )

    # Texto "LIQUID" si hay algún líquido
    if has_any_liquid:
        pos = LIQUID_TEXT_POS.get(float(cao_pct), LIQUID_TEXT_DEFAULT)
        tax.text(pos[0], pos[1], "Liquid", transform=tax.transAxes,
                 ha="center", va="center", fontsize=12, clip_on=False)

    # Leyenda (lista de zonas -> fases). Sin agrupar.
    if cao_pct == 30.0:
        # Petición usuario: solo texto abajo "Liquid + MgO + (Fe,Mg)O" y quitar cuadro
        tax.text(
            0.5, 0.2,
            "Liquid + MgO + (Fe,Mg)O",
            transform=tax.transAxes,
            ha="center", va="top", fontsize=10, color="black"
        )
    elif zone_order:
        legend_pos = ZONE_LEGEND_POS.get(float(cao_pct), (0.02, 0.98))
        lines = []
        for zone_key in zone_order:
            zid = zone_map[zone_key]
            parts = zone_label_parts.get(zid, [])
            if not parts:
                label = zone_key
            else:
                final_parts = [p for p in parts if not p.startswith("Liquid")]
                liquids = {p for p in parts if p.startswith("Liquid")}

                if liquids == {"Liquid 1", "Liquid 2"}:
                    final_parts.append("Liquid 1")
                elif liquids == {"Liquid 1", "Liquid 3"}:
                    final_parts.append("Liquid 2")
                else:
                    final_parts.extend(liquids)
                
                label = " + ".join(sorted(final_parts))
            lines.append(f"{zid} = {label}")

        tax.text(
            legend_pos[0], legend_pos[1],
            "\n".join(lines),
            transform=tax.transAxes,
            ha="left", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "none", "edgecolor": "none"},
        )

    plt.tight_layout()

    output_path: Optional[str] = None
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


# =============================================================================
# RUNNERS
# =============================================================================

def run_diagramas(*, show: bool = True, save: bool = True) -> None:
    base_dir = get_base_dir()
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    exp_files = sorted(f for f in os.listdir(base_dir) if f.lower().endswith(".exp"))
    if not exp_files:
        print("No se encontraron archivos .exp en la carpeta del script.")
        return

    for filename in exp_files:
        exp_path = os.path.join(base_dir, filename)
        output = plot_phase_diagram(
            exp_path,
            figures_dir=figures_dir,
            al2o3_pct=AL2O3_PCT_DEFAULT,
            mno_pct=MNO_PCT_DEFAULT,
            show=show,
            save=save,
        )
        if output:
            print(f"Guardado: {output}")


if __name__ == "__main__":
    run_diagramas(show=True, save=False)
