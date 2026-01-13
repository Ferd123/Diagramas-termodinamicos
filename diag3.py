import numpy as np
import matplotlib.pyplot as plt
import re
import os
import matplotlib as mpl
from matplotlib.lines import Line2D

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

# ==========================================
# MANUAL LABELS CONFIGURATION
# ==========================================
# Modificar estos valores para ajustar posición (x, y), rotación (rot) y texto.
MANUAL_LABELS = {
    'basi1.8.exp': [
        {'text': '(Fe,Mg)O + 3CS + C₂S + Liquido', 'x': 5, 'y': 12, 'rot': 0, 'color': 'black'},
        {'text': 'Liquido', 'x': 23.36, 'y': 5.86, 'rot': 0, 'color': 'black'},
        {'text': '(Fe,Ca)O + C₂S + Liquido', 'x': 10, 'y': 2.5, 'rot': 0, 'color': 'black'},
        {'text': '(Fe,Mg)O + Liquido', 'x': 20, 'y': 10, 'rot': 0, 'color': 'black'},
    ],
    'basi2.3.exp': [
        {'text': '(Fe,Mg)O //+// C₂S //+// Liquido', 'x': 1.6, 'y': 2.10, 'rot': 0, 'color': 'black'},
        {'text': '(Mg,Ca)O + (Fe,Mg)O + C₂S + Liquido', 'x': 1.15, 'y': 7, 'rot': 90, 'color': 'black'},
        {'text': '(Fe,Ca)O //+// C₂S //+// Liquido', 'x': 17.21, 'y': 8.77, 'rot': 0, 'color': 'black'},
        {'text': 'C₂S + Liquido', 'x': 15, 'y': 2.83, 'rot': 0, 'color': 'black'},
        {'text': '(Fe,Mg)O + Liquido', 'x': 26, 'y': 8, 'rot': 0, 'color': 'black'},
        {'text': 'Liquido', 'x': 30, 'y': 4, 'rot': 0, 'color': 'black'},
    ],
    'basi2.8.exp': [
        {'text': '(Fe,Ca)O + 3CS //+// C₂S + Liquido', 'x': 2, 'y': 0.5, 'rot': 0, 'color': 'black'},
        {'text': '(Fe,Mg)O //+// C₂S + Liquido', 'x': 20, 'y': 10, 'rot': 0, 'color': 'black'},
        {'text': '(Fe,Mg)O //+// Liquido', 'x': 28, 'y': 8, 'rot': 0, 'color': 'black'},
        {'text': '(Fe,Ca)O //+// (Fe,Mg)O //+// C₂S //+// Liquido', 'x': 9.55, 'y': 6.14, 'rot': 0, 'color': 'black'},
        {'text': '(Fe,Ca)O //+// C₂S + Liquido', 'x': 10, 'y': 2.09, 'rot': 0, 'color': 'black'},
        {'text': '(Fe,Ca)O //+// (Fe,Mg)O //+// 3CS //+// C₂S //+// Liquido', 'x': 3.10, 'y': 12.58, 'rot': 0, 'color': 'black'},
        {'text': '(Fe,Mg)O //+// 3CS //+// C₂S //+// Liquido', 'x': 6, 'y': 15.56, 'rot': 0, 'color': 'black'},
        {'text': 'C₂S + Liquido', 'x': 25, 'y': 2, 'rot': 0, 'color': 'black'},
    ]
}

# CENTROIDES DE DATOS INDUSTRIALES (Calculados K-Means = 10)
PROMEDIOS_INDUSTRIALES = [
    {'FeO': 24.58, 'SiO2': 21.13, 'CaO': 33.08, 'MgO': 11.06, 'Al2O3': 4.39, 'MnO': 0.00, 'label': 'Ind. 25%FeO'},
    {'FeO': 40.50, 'SiO2': 16.57, 'CaO': 26.50, 'MgO': 8.55, 'Al2O3': 3.54, 'MnO': 0.00, 'label': 'Ind. 41%FeO'},
    {'FeO': 32.52, 'SiO2': 18.82, 'CaO': 29.49, 'MgO': 9.97, 'Al2O3': 3.98, 'MnO': 0.00, 'label': 'Ind. 33%FeO'},
    {'FeO': 27.90, 'SiO2': 20.37, 'CaO': 29.56, 'MgO': 11.31, 'Al2O3': 4.27, 'MnO': 0.00, 'label': 'Ind. 28%FeO'},
    {'FeO': 0.74, 'SiO2': 14.99, 'CaO': 63.74, 'MgO': 4.31, 'Al2O3': 13.84, 'MnO': 0.00, 'label': 'Ind. 1%FeO'},
    {'FeO': 19.38, 'SiO2': 22.51, 'CaO': 34.46, 'MgO': 11.94, 'Al2O3': 4.68, 'MnO': 0.00, 'label': 'Ind. 19%FeO'},
    {'FeO': 36.29, 'SiO2': 17.21, 'CaO': 28.61, 'MgO': 9.10, 'Al2O3': 3.71, 'MnO': 0.00, 'label': 'Ind. 36%FeO'},
    {'FeO': 46.47, 'SiO2': 14.43, 'CaO': 23.91, 'MgO': 7.27, 'Al2O3': 3.15, 'MnO': 0.00, 'label': 'Ind. 46%FeO'},
    {'FeO': 28.94, 'SiO2': 18.95, 'CaO': 32.59, 'MgO': 10.03, 'Al2O3': 4.03, 'MnO': 0.00, 'label': 'Ind. 29%FeO'},
]

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

def filtrar_por_basicidad(puntos, target_B, tol=0.25):
    """
    Filtra puntos cuya basicidad (CaO/SiO2) este dentro del rango target +/- tol.
    """
    ref = []
    others = []
    for p in puntos:
        if p["SiO2"] == 0: continue # Evitar division por cero
        
        B = p["CaO"] / p["SiO2"]
        if abs(B - target_B) <= tol:
            ref.append(p)
        else:
            others.append(p)
    return ref, others

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

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================

def run_diag3():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Archivos a procesar
    files_to_process = ['basi1.8.exp', 'basi2.3.exp', 'basi2.8.exp']
    colors_pure = ['black', 'blue', 'green', 'purple', 'brown', 'orange', 'gray', 'olive', 'cyan']

    for filename in files_to_process:
        print(f"\nProcesando archivo: {filename}...")
        file_path = os.path.join(script_dir, filename)

        if not os.path.exists(file_path):
            print(f"Advertencia: Archivo no encontrado {filename}. Saltando.")
            continue
            
        # Extraer basicidad target del nombre de archivo (ej: basi1.8.exp -> 1.8)
        try:
            target_B_str = re.findall(r"basi(\d+\.\d+)", filename)[0]
            target_B = float(target_B_str)
            print(f"Basicidad objetivo detectada: {target_B}")
        except:
            print("No se pudo detectar basicidad del nombre, usando default 2.0")
            target_B = 2.0

        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception as e:
            print(f"Error leyendo {filename}: {e}")
            continue

        blocks = parse_thermocalc_exp(content)
        print(f"Bloques encontrados: {len(blocks)}")

        # Setup Figure
        width = DOUBLE_COLUMN_MM * MM_TO_INCH
        height = width
        fig, ax = plt.subplots(figsize=(width, height))

        # Estructura para agrupar segmentos por fase (para Auto-Labeling)
        phases_data = {} 

        # 1. Graficar Fases
        for i, block in enumerate(blocks):
            phases_set = set(block["phases"])
            # Filtros de fases liquidas especificas
            # "si hay una linea exclusiva de esas dos fases la elimines"
            if phases_set == {'IONIC_LIQ#1', 'IONIC_LIQ#2'}: continue
            # El usuario no especifico quitar la otra (L1+L2+Halite#2), pero en general L1+L2 boundary se quita.
            # Mantendremos la anterior por seguridad visual si molesta.
            if phases_set == {'IONIC_LIQ#1', 'IONIC_LIQ#2','HALITE#2'}: continue
            
            # Identificador unico para el conjunto de fases
            phases_key = tuple(sorted(list(phases_set)))
            if phases_key not in phases_data:
                phases_data[phases_key] = []
            
            # Color hashing
            color_seed = sum(map(ord, "".join(phases_key))) 
            c = colors_pure[color_seed % len(colors_pure)]
            
            for segment in block["segments"]:
                if len(segment) == 0: continue
                
                # Check magnitude of first point for scaling logic
                if np.max(segment) <= 1.0:
                     seg_scaled = segment * 100.0
                else:
                     seg_scaled = segment
                     
                ax.plot(seg_scaled[:, 0], seg_scaled[:, 1], color=c, linewidth=0.8)
                
                # Guardar datos para centroid labeling
                phases_data[phases_key].append(seg_scaled)

        # 2. Labels Manuales (Desde Diccionario para edición manual)
        current_labels = MANUAL_LABELS.get(filename, [])
        for lbl_config in current_labels:
            # Support for multi-line labels using '//' separator
            final_text = lbl_config['text'].replace('//', '\n')
            ax.text(lbl_config['x'], lbl_config['y'], final_text, 
                    fontsize=6, color=lbl_config['color'], 
                    ha='center', va='center', rotation=lbl_config['rot'],
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.1))

        # 3. Puntos EAF (Actualizado para FeO vs MgO y Filtro Basicidad)
        eaf_points_all = puntos_EAF_completos()
        # Ampliar rango valido para MgO
        valid_points = [p for p in eaf_points_all if 0 <= p['FeO'] <= 40 and 0 <= p['MgO'] <= 40]
        
        # Filtro por Basicidad
        points_ref, points_other = filtrar_por_basicidad(valid_points, target_B, tol=0.25)

        if points_other:
            other_x = [p['FeO'] for p in points_other]
            other_y = [p['MgO'] for p in points_other] 
            ax.scatter(other_x, other_y, c='blue', edgecolors='black', linewidth=0.5, marker='o', s=15, zorder=5, label='Other Basicity')
            # Opcional: Labels para 'others' pueden ensuciar mucho grafico, comentar si es necesario
            # for p in points_other:
            #    ax.text(p['FeO'] - 0.2, p['MgO'], p['label'], fontsize=4, color='darkblue', ha='right', va='center')

        if points_ref:
            ref_x = [p['FeO'] for p in points_ref]
            ref_y = [p['MgO'] for p in points_ref] 
            ax.scatter(ref_x, ref_y, c='red', edgecolors='black', linewidth=0.5, marker='o', s=20, zorder=6, label=f'Ref B={target_B}$\\pm$0.25')
            for p in points_ref:
                ax.text(p['FeO'] - 0.2, p['MgO'], p['label'], fontsize=5, color='darkred', ha='right', va='center', rotation=0, zorder=7)

        # 4. Puntos Industriales (Nuevos)
        # Filtrar rango valido (0-40 FeO)
        valid_ind = [p for p in PROMEDIOS_INDUSTRIALES if 0 <= p['FeO'] <= 40]
        ind_ref, ind_others = filtrar_por_basicidad(valid_ind, target_B, tol=0.25)
        
        # Plot matches (Green Squares)
        if ind_ref:
            ix = [p['FeO'] for p in ind_ref]
            iy = [p['MgO'] for p in ind_ref]
            ax.scatter(ix, iy, c='green', edgecolors='black', linewidth=0.5, marker='s', s=25, zorder=8, label=f'Ind. Match')
            for p in ind_ref:
                # Add white background to label for visibility
                ax.text(p['FeO'], p['MgO']+0.5, p['label'], fontsize=5, color='darkgreen', ha='center', va='bottom', weight='bold',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.1))
        
        # Plot others (Cyan Squares - Faded)
        if ind_others:
            io_x = [p['FeO'] for p in ind_others]
            io_y = [p['MgO'] for p in ind_others]
            ax.scatter(io_x, io_y, c='cyan', edgecolors='black', linewidth=0.5, marker='s', s=20, zorder=7, label=f'Ind. Other', alpha=0.6)
            # No labels for non-matching industrial points to reduce clutter

        # Configuración final
        title_str = filename.replace('.exp', '').replace('basi', 'Basicidad ').capitalize()
        ax.set_title(f"Diagrama de Fases: {title_str}", fontsize=14)
        ax.set_xlabel("Mass percent FeO", fontsize=12)
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 17.50)
        ax.set_ylabel("Mass percent MgO", fontsize=12) # CAMBIO Y Label
        ax.set_xlim(0, 40) # CAMBIO Limites
        ax.set_ylim(0, 17.50) # MgO suele ser menor, pero pondremos 20 o 40 segun datos. Header dice YSCALE ~18.
        ax.grid(True, linestyle=':', alpha=0.4)

        plt.tight_layout()
        output_path = os.path.join(figures_dir, f'diagrama_{filename.replace(".exp", "")}.png')
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Guardado: {output_path}")
        plt.close(fig)

if __name__ == "__main__":
    run_diag3()
