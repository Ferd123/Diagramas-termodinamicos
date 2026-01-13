import numpy as np
import matplotlib.pyplot as plt
import re
import os
import matplotlib as mpl

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
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
})

# ==========================================
# 1. HELPERS: PARSING & DATA
# ==========================================

def parse_thermocalc_exp(content):
    blocks = []
    current_block = None
    metadata = {
        "xscale": None,
        "yscale": None
    }
    
    number_pattern = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Metadata parsing (Header)
        if line.startswith("XSCALE"):
            nums = [float(x) for x in number_pattern.findall(line)]
            if len(nums) >= 2: metadata["xscale"] = nums
        elif line.startswith("YSCALE"):
            nums = [float(x) for x in number_pattern.findall(line)]
            if len(nums) >= 2: metadata["yscale"] = nums

        if line.startswith("$ BLOCK") or line.startswith("$BLOCK"):
            if current_block: blocks.append(current_block)
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
                numbers = [float(x) for x in number_pattern.findall(line)]
                if len(numbers) >= 2:
                    x_val, y_val = numbers[0], numbers[1]
                    if 'M' in line:
                        if current_block["current_segment"]:
                            current_block["segments"].append(np.array(current_block["current_segment"]))
                            current_block["current_segment"] = []
                        current_block["current_segment"].append([x_val, y_val])
                    else:
                        current_block["current_segment"].append([x_val, y_val])

    if current_block:
        if current_block["current_segment"]:
            current_block["segments"].append(np.array(current_block["current_segment"]))
        blocks.append(current_block)
    return blocks, metadata

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

def clasificar_puntos_mgo(puntos, diagramas_objetivo=[3, 7, 10, 15]):
    grupos = {k: [] for k in diagramas_objetivo}
    for punto in puntos:
        mgo_real = punto["MgO"]
        diagrama_cercano = min(diagramas_objetivo, key=lambda x: abs(x - mgo_real))
        grupos[diagrama_cercano].append(punto)
    return grupos

def filtrar_referencia_5_5_7(puntos, Al2O3_target=5.0, Al2O3_tol=2.0, MgO_target=7.0, MgO_tol=2.0):
    filtrados = []
    for p in puntos:
        if (Al2O3_target - Al2O3_tol <= p["Al2O3"] <= Al2O3_target + Al2O3_tol and
            MgO_target  - MgO_tol  <= p["MgO"]   <= MgO_target  + MgO_tol):
            filtrados.append(p)
    return filtrados

# ==========================================
# 2. PLOTTING FUNCTION
# ==========================================

def generate_diagram(exp_filename, title, output_filename, points_list, is_pure=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    exp_path = os.path.join(script_dir, exp_filename)
    print(f"\nGenerando: {title} ({exp_filename})")
    print(f"Puntos a graficar: {len(points_list)}")

    try:
        with open(exp_path, 'r', encoding='latin-1') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: {exp_filename} no encontrado.")
        return

    blocks, metadata = parse_thermocalc_exp(content)
    
    # Auto-Scaling Logic
    scale_factor = 1.0
    x_max = 34.0 # Default fallback
    y_max = 34.0
    
    if metadata["xscale"]:
        x_min_file, x_max_file = metadata["xscale"]
        if x_max_file <= 1.0: # Heuristic: If max <= 1.0, assume Fraction -> Convert to %
            scale_factor = 100.0
            print(f"   -> Detectado Formato Fraccional (Max X={x_max_file}). Aplicando escala x100.")
        else:
            print(f"   -> Detectado Formato Porcentual (Max X={x_max_file}). Escala x1.")
            
        x_max = x_max_file * scale_factor
    
    if metadata["yscale"]:
        y_min_file, y_max_file = metadata["yscale"]
        y_max = y_max_file * scale_factor

    # Setup Figure
    width = DOUBLE_COLUMN_MM * MM_TO_INCH
    height = width
    fig, ax = plt.subplots(figsize=(width, height))
    
    colors_cycle = ['black', 'blue', 'green', 'purple', 'brown', 'orange', 'gray', 'olive', 'cyan']
    
    for i, block in enumerate(blocks):
        phases_set = set(block["phases"])
        if phases_set == {'IONIC_LIQ#1', 'IONIC_LIQ#2'}: continue
        if phases_set == {'IONIC_LIQ#1', 'IONIC_LIQ#2','HALITE#2'}: continue
        
        color_seed = sum(map(ord, "".join(sorted(block["phases"])))) 
        c = colors_cycle[color_seed % len(colors_cycle)]
        
        for segment in block["segments"]:
            if len(segment) == 0: continue
            seg_scaled = segment * scale_factor
            ax.plot(seg_scaled[:, 0], seg_scaled[:, 1], color=c, linewidth=0.8)

    # Specific Labels for PURE diagram
    if is_pure:
        ax.text(5.89, 13.16, 'Liquido + (Mn,Fe)O + (Fe,Ca)O + 3CS', rotation=0, fontsize=8, color='brown', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
        ax.text(10, 15.4, 'Liquido + (Fe,Ca)O + 3CS', rotation=-25, fontsize=4, color='green', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
        ax.text(5.88, 18.70, 'Liquido + C₂S + (Fe,Ca),O + 3CS', rotation=-28, fontsize=8, color='brown', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
        ax.text(8, 21.5, 'Liquido + C₂S + (Fe,Ca)O', rotation=-32, fontsize=8, color='blue', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
        ax.text(8.21, 26, 'Liquido + C₂S', rotation=-30, fontsize=8, color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
        ax.text(24.41, 10, 'Liquido + (Fe,Ca)O', rotation=-30, fontsize=8, color='blue', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
        ax.text(15.64, 5.87, 'Liquido + (Fe,Ca)O + (Mn,Fe)O', rotation=0, fontsize=8, color='blue', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
        ax.text(25,20, 'Liquido', rotation=0, fontsize=12, color='blue', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))

    # Plot Points
    ref_candidates = filtrar_referencia_5_5_7(points_list)
    
    valid_points = [p for p in points_list if 0 <= p['FeO'] <= x_max and 0 <= p['SiO2'] <= y_max]
    
    points_ref = []
    points_other = []
    
    for p in valid_points:
        if p in ref_candidates:
            points_ref.append(p)
        else:
            points_other.append(p)
            
    if points_other:
        other_x = [p['FeO'] for p in points_other]
        other_y = [p['SiO2'] for p in points_other]
        ax.scatter(other_x, other_y, c='blue', edgecolors='black', linewidth=0.5, marker='o', s=15, zorder=5, label='Otros EAF')
        for p in points_other:
            ax.text(p['FeO'] - 0.2, p['SiO2'], p['label'], fontsize=5, color='darkblue', ha='right', va='center', zorder=6)

    if points_ref:
        ref_x = [p['FeO'] for p in points_ref]
        ref_y = [p['SiO2'] for p in points_ref]
        ax.scatter(ref_x, ref_y, c='red', edgecolors='black', linewidth=0.5, marker='o', s=20, zorder=6, label='EAF Ref (5%Al/7%Mg)')
        for p in points_ref:
            ax.text(p['FeO'] - 0.2, p['SiO2'], p['label'], fontsize=5, color='darkred', ha='right', va='center', zorder=7)

    # Basicity Lines
    basicities = [1.0, 2.0, 2.5, 3.0]
    x_fe_range = np.linspace(0, x_max, 100)
    for B in basicities:
        y_si_vals = (83 - x_fe_range) / (B + 1)
        mask = (y_si_vals >= 0) & (y_si_vals <= y_max)
        if not np.any(mask): continue
        
        x_plot = x_fe_range[mask]
        y_plot = y_si_vals[mask]
        
        ax.plot(x_plot, y_plot, color='black', linestyle='--', linewidth=0.8, alpha=0.7, zorder=2)
        
        angle_rad = np.arctan(-1 / (B + 1))
        angle_deg = np.degrees(angle_rad)
        
        lbl_idx = 0 
        if B == 1.0: lbl_idx = int(len(x_plot) * 0.4)
        
        lbl_x = x_plot[lbl_idx]
        lbl_y = y_plot[lbl_idx]
        
        ax.text(lbl_x + 0.5, lbl_y + 0.5, f"B={B}", fontsize=7, color='black', ha='left', va='bottom', rotation=angle_deg, zorder=3)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Mass percent FeO", fontsize=12)
    ax.set_ylabel("Mass percent SiO$_2$", fontsize=12)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    output_path = os.path.join(figures_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Guardado: {output_path}")
    plt.close(fig)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    
    all_points = puntos_EAF_completos()
    groups = clasificar_puntos_mgo(all_points)
    
    # 1. 3% Diagram 
    generate_diagram(
        exp_filename="3PERCENT.exp",
        title="Diagrama de Fases (3% MgO)",
        output_filename="diagrama_3mgo.png",
        points_list=groups[3],
        is_pure=False
    )
    
    # 2. Pure Diagram (~7% MgO) 
    generate_diagram(
        exp_filename="diagrama fases.exp",
        title="Diagrama de Fases Puro (~7% MgO)",
        output_filename="diagrama_fases_puro.png",
        points_list=groups[7],
        is_pure=True
    )
    
    # 3. 10% Diagram 
    generate_diagram(
        exp_filename="10porciento.exp",
        title="Diagrama de Fases (10% MgO)",
        output_filename="diagrama_10mgo.png",
        points_list=groups[10],
        is_pure=False
    )
    
    # 4. 15% Diagram 
    generate_diagram(
        exp_filename="15porciento.exp",
        title="Diagrama de Fases (15% MgO)",
        output_filename="diagrama_15mgo.png",
        points_list=groups[15],
        is_pure=False
    )
    
    print("\nProceso terminado.")
