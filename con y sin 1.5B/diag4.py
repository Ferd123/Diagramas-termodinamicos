import numpy as np
import matplotlib.pyplot as plt
import re
import os
import glob
import matplotlib as mpl
import pandas as pd # Needed for excel

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
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "svg.fonttype": "none",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
})
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
# ==========================================
# PHASE RENAMING
# ==========================================
PHASE_MAPPING = {
    'HALITE': '(Fe,Mg)O',
    'HALITE#2': '(Fe,Mg)O',
    'CA2SIO4_ALPHA_A': "α-C₂S",
    'CA2SIO4_ALPHA_PRIME': "α'-C₂S",
    'CA2SIO4_BETA': 'C₂S',
    'IONIC_LIQ': 'Liquido',
    'IONIC_LIQ#1': 'Liquido',
    'IONIC_LIQ#2': 'Liquido',
    'CA3SIO5_H': 'C₃S',
    'CA3SIO5_M': 'C₃S',
    'CA3SIO5_L': 'C₃S',
    'LIME': '(Ca,Fe)O', # Often just 'Lime' or '(Ca,Fe)O' depending on context
    'PERICLASE': 'Periclase',
}

def clean_phase_name(raw_name):
    """
    Cleans up Thermo-Calc phase names to be more readable.
    """
    clean = raw_name.strip()
    
    # Check exact matches first
    if clean in PHASE_MAPPING:
        return PHASE_MAPPING[clean]
        
    # Heuristic for generic cases (e.g., removing #1, #2 suffix if not mapped)
    base_name = clean.split('#')[0]
    if base_name in PHASE_MAPPING:
        return PHASE_MAPPING[base_name]
        
    # Special naming rules
    if "IONIC_LIQ" in clean: return "Liquido"
    if "HALITE" in clean: return "(Fe,Mg)O"
    if "CA2SIO4" in clean: return "C₂S"
    
    return clean

# ==========================================
# LABEL ADJUSTMENTS
# ==========================================
LABEL_OFFSETS = {
    "Liquido + α'-C₂S": (-5, 0),    # Move Left
    "(Fe,Mg)O + Liquido": (0, 2),   # Move Up
}

SPECIAL_CONFIG = {
    "1.5FeOCaOSiO2Al2O3MnOMgO.exp": {
        "Liquido + α-C₂S": {
            "text": "α-C₂S\n+\nLiquido", # Multiline and reordered
            "style": "simple",
            "force_x": 2.5
        },
        "(Fe,Mg)O + Liquido + α-C₂S": {
             "pos": (5, 16),
             "style": "arrow"
        }
    }
}

def format_label(phases):
    """
    Creates a combined label for a list of phases.
    Sorts them to be consistent.
    """
    unique_names = set()
    for p in phases:
        unique_names.add(clean_phase_name(p))
    
    # Sort for consistency
    sorted_names = sorted(list(unique_names))
    return " + ".join(sorted_names)


# ==========================================
# PARSER
# ==========================================
def parse_thermocalc_exp(content):
    blocks = []
    current_block = None
    
    number_pattern = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line: continue

        if line.startswith("$ BLOCK") or line.startswith("$BLOCK"):
            if current_block: blocks.append(current_block)
            current_block = {
                "phases": [],
                "segments": [],
                "current_segment": []
            }
        
        elif line.startswith("$F0") or line.startswith("$E"):
            if current_block:
                parts = line.split(maxsplit=1)
                phase_name = parts[1] if len(parts) > 1 else "Unknown"
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
                    x_val = numbers[0] 
                    y_val = numbers[1]
                    
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
        
    return blocks

def filtrar_por_basicidad(puntos, target_B, tol=0.25):
    """
    Filtra puntos cuya basicidad (CaO/SiO2) este dentro del rango target +/- tol.
    """
    ref = []
    others = []
    for p in puntos:
        # Check dictionary vs obj, user might pass different structs
        # Standardization:
        sio2 = p.get('SiO2', 0)
        cao = p.get('CaO', 0)
        
        # If NaN or 0
        if pd.isna(sio2) or sio2 == 0: continue
        
        B = cao / sio2
        if abs(B - target_B) <= tol:
            ref.append(p)
        else:
            others.append(p)
    return ref, others

def leer_datos_excel_mittal():
    """Reads Industrial Data from Excel"""
    # Assuming relative path or fixed path based on user usage in other files
    # User edits in datosk.py suggest: r"C:\Users\ELANOR\Documents\ARCELORMITTAL\datos mittal.xlsx"
    path = r"C:\Users\ELANOR\Documents\ARCELORMITTAL\datos mittal.xlsx"
    if not os.path.exists(path):
        print(f"Excel no encontrado: {path}")
        return []
        
    try:
        df = pd.read_excel(path, sheet_name="Sheet1")
        
        # MAPPING COLUMNS based on debug output
        # HO_FeO, HO_SiO2, HO_CaO, HO_MgO, HO_Al2O3, HO_Mno
        rename_map = {
            'HO_FeO': 'FeO',
            'HO_SiO2': 'SiO2',
            'HO_CaO': 'CaO',
            'HO_MgO': 'MgO',
            'HO_Al2O3': 'Al2O3',
            'HO_Mno': 'MnO',
            'Ho_Bas': 'Basicity' # Optional
        }
        
        # Check if columns exist before renaming to avoid errors if file changes
        # Filter only cols that exist
        actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
        
        if not actual_rename:
            print("No se encontraron columnas de composicón (HO_...) en el excel.")
            return []

        df = df.rename(columns=actual_rename)
        
        # Drop rows with NaN in critical cols
        critical_cols = ['FeO', 'SiO2', 'CaO', 'MgO']
        # Filter cols that are actually in df now
        existing_crit = [c for c in critical_cols if c in df.columns]
        
        if existing_crit:
            df = df.dropna(subset=existing_crit)
        
        records = df.to_dict('records')
        # Add a label info if missing?
        for r in records:
            if 'label' not in r: r['label'] = 'Ind'
            
        return records
    except Exception as e:
        print(f"Error leyendo excel: {e}")
        return []

# ==========================================
# ANALYSIS HELPER
# ==========================================
def calculate_centroid(segments):
    """
    Calculates the centroid (average X, Y) of all points in all segments.
    """
    all_x = []
    all_y = []
    for seg in segments:
        all_x.extend(seg[:, 0])
        all_y.extend(seg[:, 1])
    
    if not all_x: return None, None
    return np.mean(all_x), np.mean(all_y)

def format_chem_title(filename):
    """
    Formats filename into a nice chemical title with subscripts.
    Ex: FeOCaOSiO2MgO -> FeO - CaO - SiO$_2$ - MgO
    """
    name = filename.replace('.exp', '')
    
    # Common Oxides Map
    subs = [
        ('SiO2', 'SiO$_2$'),
        ('Al2O3', 'Al$_2$O$_3$'),
        ('Fe2O3', 'Fe$_2$O$_3$'),
        ('Cr2O3', 'Cr$_2$O$_3$'),
        ('TiO2', 'TiO$_2$'),
        ('MgO', 'MgO'),
        ('CaO', 'CaO'),
        ('FeO', 'FeO'),
        ('MnO', 'MnO'),
    ]
    
    # Simple strategy: Just replace knowns. 
    # Since they are concatenated, we might need to split them first if possible, 
    # but simple replacement might work if order is careful (Longer first).
    
    # Formatting for specific filename structure "FeOCaOSiO2MgO"
    # Try inserting spaces/dashes between oxides if they are stuck together?
    # Heuristic: split before capital letter? No, 'SiO2' has caps in middle.
    
    # Let's iterate and replace, and join with dashes if needed.
    # Actually, let's just do a greedy replacement and add separators.
    
    formatted = name
    for plain, latex in subs:
        if plain in formatted:
            formatted = formatted.replace(plain, latex + '-') # Tempo separator
            
    # Clean up trailing dash
    if formatted.endswith('-'): formatted = formatted[:-1]
    
    return formatted

# ==========================================
# MAIN
# ==========================================
def run_diag4():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all .exp files in the current directory
    exp_files = glob.glob(os.path.join(script_dir, "*.exp"))
    
    if not exp_files:
        print(f"No .exp files found in {script_dir}")
        return

    figures_dir = os.path.join(script_dir, "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    colors_pure = ['black', 'blue', 'green', 'purple', 'brown', 'orange', 'gray', 'olive', 'cyan']

    for file_path in exp_files:
        filename = os.path.basename(file_path)
        print(f"\nProcesando: {filename}...")
        
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception as e:
            print(f"Error leyendo {filename}: {e}")
            continue

        blocks = parse_thermocalc_exp(content)
        print(f"Bloques encontrados: {len(blocks)}")

        # Prepare Plot
        width = DOUBLE_COLUMN_MM * MM_TO_INCH
        height = width * 0.8  # slightly shorter
        fig, ax = plt.subplots(figsize=(width, height))

        # Data structure for labeling: Map "Label String" -> [List of all segments for that label]
        label_groups = {}

        # 1. Plot Phase Boundaries
        for block in blocks:
            phases_set = set(block["phases"])
            
            # Filters
            if phases_set == {'IONIC_LIQ#1', 'IONIC_LIQ#2'}: continue # Liquid miscibility gap often hidden
             # Keep HALITE#2 based on user previous preferences, or hide if it causes issues.
            
            # Generate key and label
            label_str = format_label(block["phases"])
            
            # Store for centroid calculation
            if label_str not in label_groups:
                label_groups[label_str] = []
            
            # Plot segments
            # Use deterministic color based on sorted phases keys
            phases_key_sorted = tuple(sorted(list(phases_set)))
            color_seed = sum(map(ord, "".join(phases_key_sorted))) 
            c = colors_pure[color_seed % len(colors_pure)]

            for segment in block["segments"]:
                if len(segment) == 0: continue
                
                # Check scaling (if data is 0-1, scale to 0-100)
                # Assuming standard MgO/FeO plot which is usually 0-100 range values in file or 0-1
                # The provided file seemed to have values like 5.9, so 0-100 scale is likely already applied or its small %
                # Let's check max value to decide scaling.
                max_val = np.max(segment)
                if max_val <= 1.0:
                     seg_scaled = segment * 100.0
                else:
                     seg_scaled = segment
                
                ax.plot(seg_scaled[:, 0], seg_scaled[:, 1], color=c, linewidth=0.8)
                label_groups[label_str].append(seg_scaled)

        # 2. Auto-Labeling
        print("Generando labels automaticos...")
        existing_labels_pos = [] # To avoid overlap (simple check)

        for label_text, segments in label_groups.items():
            if not segments: continue
            
            # FILTRO USUARIO: Eliminar etiqueta donde conviven las dos fases alpha
            if "α-C₂S" in label_text and "α'-C₂S" in label_text:
                continue

            cx, cy = calculate_centroid(segments)
            if cx is None: continue
            
            # Check for Special Config
            config = SPECIAL_CONFIG.get(filename, {}).get(label_text, None)
            
            if config:
                if config["style"] == "arrow":
                    # Annotation with arrow
                    # Tip of arrow = Calculated Centroid (cx, cy)
                    # Text Position = Specified pos
                    tx, ty = config["pos"]
                    ax.annotate(label_text, xy=(cx, cy), xytext=(tx, ty),
                                arrowprops=dict(facecolor='black', arrowstyle='->', shrinkA=0, shrinkB=0, lw=0.8),
                                fontsize=6, ha='center', va='center',
                                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=0.1))
                    continue # Skip normal text
                
                elif config["style"] == "simple":
                    # Just change text content (e.g. multiline)
                    label_text = config["text"]
                    if "force_x" in config:
                        cx = config["force_x"]
                    # Continues to offset logic below...
            
            # Apply Manual Offsets
            if label_text in LABEL_OFFSETS:
                dx, dy = LABEL_OFFSETS[label_text]
                cx += dx
                cy += dy

            # Simple collision avoidance (very basic)
            # If too close to existing label, maybe skip? 
            # For now, just plot and let user refine.
            
            # Color logic: black text with white box
            ax.text(cx, cy, label_text, fontsize=6, color='black', 
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.1))
        

        
        # 3. Etiquetas Manuales Adicionales (Solicitud Usuario)
        ax.text(37, 4, "Liquido", fontsize=6, color='black', 
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.1))

        # ---------------------------------------------------------
        # 4. PLOTTING POINTS (EAF & Industrial)
        # ---------------------------------------------------------
        
        # Determine Target Basicity
        # User explicitly set B=1.5 in title text manually, so we assume 1.5 is the filter target.
        # Or we can try to parse from filename if present (e.g. 1.5FeOCaOSiO2...)
        # FileName: "1.5FeOCaOSiO2Al2O3MnOMgO.exp" starts with 1.5
        target_B = 1.5 
        try:
            # Try to grab leading float
            match = re.match(r"(\d+(\.\d+)?)", filename)
            if match:
                target_B = float(match.group(1))
                print(f"Basicidad detectada de nombre archivo: {target_B}")
        except:
            pass

        # A) EAF POINTS
        eaf_points_all = puntos_EAF_completos()
        # Filter valid range
        valid_points = [p for p in eaf_points_all if 0 <= p['FeO'] <= 40 and 0 <= p['MgO'] <= 40]
        points_ref, points_other = filtrar_por_basicidad(valid_points, target_B, tol=0.25)

        # Plot "Other" EAF (Blue)
        if points_other:
            other_x = [p['FeO'] for p in points_other]
            other_y = [p['MgO'] for p in points_other] 
            # ax.scatter(other_x, other_y, c='blue', edgecolors='black', linewidth=0.5, marker='o', s=15, zorder=5, label='Other Basicity')

        # Plot "Ref" EAF (Red)
        if points_ref:
            ref_x = [p['FeO'] for p in points_ref]
            ref_y = [p['MgO'] for p in points_ref] 
            ax.scatter(ref_x, ref_y, c='red', edgecolors='black', linewidth=0.5, marker='o', s=20, zorder=6, label=f'Ref B={target_B}$\\pm$0.25')
            for p in points_ref:
                ax.text(p['FeO'] - 0.2, p['MgO'], p['label'], fontsize=5, color='darkred', ha='right', va='center', rotation=0, zorder=7)

        # B) INDUSTRIAL POINTS (FROM EXCEL)
        ind_data = leer_datos_excel_mittal()
        if ind_data:
            # Filter valid range
            valid_ind = [p for p in ind_data if 0 <= p.get('FeO', -1) <= 40]
            ind_ref, ind_others = filtrar_por_basicidad(valid_ind, target_B, tol=0.25)
            print(f"Puntos Industriales encontrados: {len(valid_ind)}, Match B={target_B}: {len(ind_ref)}")
            
            # --- Logic for Average and Representative Points ---
            if ind_ref:
                # 1. Average
                avg_FeO = np.mean([p['FeO'] for p in ind_ref])
                avg_MgO = np.mean([p['MgO'] for p in ind_ref])
                
                # Plot Average
                ax.scatter(avg_FeO, avg_MgO, c='gold', edgecolors='black', marker='*', s=150, zorder=10, label=f'Promedio Ind.')
                ax.text(avg_FeO, avg_MgO+0.5, "Promedio", fontsize=6, color='black', ha='center', va='bottom', weight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.1))

                # 2. Representative Points (Max 10)
                # Strategy: Sort by FeO and pick equidistant to cover range
                sorted_ind = sorted(ind_ref, key=lambda k: k['FeO'])
                total_pts = len(sorted_ind)
                
                if total_pts <= 10:
                    rep_points = sorted_ind
                else:
                    indices = np.linspace(0, total_pts-1, 10, dtype=int)
                    rep_points = [sorted_ind[i] for i in indices]
                
                # Plot Representative
                ix = [p['FeO'] for p in rep_points]
                iy = [p['MgO'] for p in rep_points]
                ax.scatter(ix, iy, c='green', edgecolors='black', linewidth=0.5, marker='s', s=30, zorder=8, label=f'Ind. Rep. (10)')
                 
                 # Optional: Verify Basicity of these points in console
                print("\nVerificación Basicidad Puntos Representativos:")
                for i, p in enumerate(rep_points):
                     b_real = p['CaO']/p['SiO2'] if p['SiO2']!=0 else 0
                     print(f"Pt {i+1}: FeO={p['FeO']:.1f}, B={b_real:.2f}")

            # Plot Others (Cyan Squares) - Maybe hide if too many? User asked for 10 rep points, implies cleanliness.
            # Keeping it faint or removing. Let's keep it very faint to show population context if desired, or remove if clutter.
            # User phrase: "saca un promedio y 10 puntos representativos" -> heavily implies ONLY plot those.
            # I will comment out the 'Others' plotting to clean up the plot as requested implicitly.
            # if ind_others:
            #     io_x = [p['FeO'] for p in ind_others]
            #     io_y = [p['MgO'] for p in ind_others]
            #     ax.scatter(io_x, io_y, c='cyan', edgecolors='black', linewidth=0.5, marker='s', s=20, zorder=7, label=f'Ind. Other', alpha=0.2)

        # ---------------------------------------------------------
        # 5. Formatting
        title_str = format_chem_title(filename)
        ax.set_title(f"Diagrama de Fases: {title_str} B=1.5", fontsize=12)
        ax.set_xlabel("Mass percent FeO", fontsize=10)
        ax.set_ylabel("Mass percent MgO", fontsize=10)
        
        # Set limits based on data or standard Fixed limits?
        # User had 0-40 FeO and 0-17.5 MgO in diag3. Let's try to stick to that but dynamic if needed.
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 17.5)
        
        ax.grid(True, linestyle=':', alpha=0.4)
        
        plt.tight_layout()
        output_path = os.path.join(figures_dir, f'diagrama_{filename.replace(".exp", "")}.png')
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Guardado: {output_path}")
        plt.close(fig)

if __name__ == "__main__":
    run_diag4()
