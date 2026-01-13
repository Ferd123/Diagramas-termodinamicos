import numpy as np
import matplotlib.pyplot as plt
import re
import os
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import splprep, splev
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
    "svg.fonttype": "none",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
})
def puntos_EAF_completos():
    """
    EAF slag literature data.
    Fe treated as FeO equivalent.
    """
    return [
        {"FeO":25.0,  "SiO2":14.0, "Al2O3":5.00, "MgO":14.0, "label":"Iran"},
        {"FeO":42.4,  "SiO2":20.3, "Al2O3":7.30, "MgO":8.00, "label":"India"},
        {"FeO":33.2,  "SiO2":10.0, "Al2O3":10.18,"MgO":8.00, "label":"China"},
        {"FeO":33.3,  "SiO2":19.3, "Al2O3":9.40, "MgO":3.07, "label":"Malaysia"},
        {"FeO":36.8,  "SiO2":13.1, "Al2O3":5.51, "MgO":5.03, "label":"Egypt"},
        {"FeO":31.7,  "SiO2":21.7, "Al2O3":8.83, "MgO":2.60, "label":"Malaysia"},
        {"FeO":37.5,  "SiO2":9.71, "Al2O3":8.21, "MgO":2.17, "label":"Italy"},
        {"FeO":43.4,  "SiO2":26.4, "Al2O3":4.84, "MgO":1.86, "label":"Malaysia-CarbonSteel"},
        {"FeO":33.3,  "SiO2":20.8, "Al2O3":9.19, "MgO":2.06, "label":"Malaysia"},
        {"FeO":0.54,  "SiO2":34.7, "Al2O3":6.26, "MgO":9.06, "label":"France-StainlessSteel"},
        {"FeO":35.0,  "SiO2":14.0, "Al2O3":12.0, "MgO":5.00, "label":"Italy"},
        {"FeO":26.8,  "SiO2":19.1, "Al2O3":13.7, "MgO":2.50, "label":"Spain-CarbonSteel"},
        {"FeO":22.3,  "SiO2":20.3, "Al2O3":12.2, "MgO":3.00, "label":"Spain"},
        {"FeO":22.0,  "SiO2":21.4, "Al2O3":9.60, "MgO":4.89, "label":"Malaysia"},
        {"FeO":34.7,  "SiO2":16.3, "Al2O3":8.31, "MgO":6.86, "label":"Vietnam-CarbonSteel"},
        {"FeO":7.54,  "SiO2":27.8, "Al2O3":2.74, "MgO":7.35, "label":"China-StainlessSteel"},
        {"FeO":24.5,  "SiO2":20.9, "Al2O3":12.1, "MgO":3.20, "label":"Spain"},
        {"FeO":28.6,  "SiO2":18.1, "Al2O3":5.88, "MgO":5.80, "label":"Malaysia"},
        {"FeO":43.0,  "SiO2":10.8, "Al2O3":6.86, "MgO":1.65, "label":"Malaysia"},
        {"FeO":27.3,  "SiO2":17.3, "Al2O3":4.67, "MgO":5.39, "label":"Malaysia"},
        {"FeO":25.9,  "SiO2":19.5, "Al2O3":4.88, "MgO":4.25, "label":"Iran"},
    ]

def filtrar_referencia_5_5_7(puntos,
                             Al2O3_target=5.0, Al2O3_tol=2.0,
                             MgO_target=7.0,  MgO_tol=2.0):
    """
    Filters points close to a 5% Al2O3 – 7% MgO reference slag.
    """
    filtrados = []
    for p in puntos:
        if (Al2O3_target - Al2O3_tol <= p["Al2O3"] <= Al2O3_target + Al2O3_tol and
            MgO_target  - MgO_tol  <= p["MgO"]   <= MgO_target  + MgO_tol):
            filtrados.append(p)
    return filtrados
# ==========================================
# 1. LEER ARCHIVO .EXP DIRECTAMENTE
# ==========================================
# Obtener la ruta del directorio donde está este script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Nombre del archivo .exp (debe estar en la misma carpeta)
exp_filename = "NUEVO.exp"
exp_path = os.path.join(script_dir, exp_filename)

# Crear carpeta de figuras si no existe
figures_dir = os.path.join(script_dir, "figures")
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
    print(f"Directorio creado: {figures_dir}")

print(f"Leyendo archivo: {exp_path}")

try:
    with open(exp_path, 'r', encoding='latin-1') as f:
        exp_content = f.read()
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {exp_filename}")
    print("Asegúrate de que el archivo .exp esté en la misma carpeta que este script.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al leer el archivo: {e}")
    exit()

# ==========================================
# 2. FUNCIÓN PARA LEER EL FORMATO .EXP
# ==========================================
def parse_thermocalc_exp(content):
    blocks = []
    current_block = None
    
    # Expresión regular para encontrar números (float notation)
    number_pattern = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")

    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line: continue

        # Detectar inicio de bloque o metadata
        if line.startswith("$ BLOCK") or line.startswith("$BLOCK"):
            if current_block: blocks.append(current_block)
            current_block = {
                "phases": [],
                "segments": [],
                "current_segment": []
            }
        
        elif line.startswith("$F0") or line.startswith("$E"):
            if current_block:
                # Extraer nombre de la fase (quitando $F0 o $E)
                phase_name = line.split(maxsplit=1)[1] if len(line.split()) > 1 else "Unknown"
                current_block["phases"].append(phase_name)

        elif line.startswith("BLOCKEND"):
            if current_block:
                # Guardar el último segmento si existe
                if current_block["current_segment"]:
                    current_block["segments"].append(np.array(current_block["current_segment"]))
                blocks.append(current_block)
                current_block = None

        elif line.startswith("BLOCK") or line.startswith("$"):
            # Ignorar líneas de configuración BLOCK X=... o comentarios
            pass
            
        else:
            # Es una línea de DATOS
            if current_block:
                # Buscar números en la línea
                numbers = [float(x) for x in number_pattern.findall(line)]
                
                if len(numbers) >= 2:
                    # Multiplicamos por 100 AQUÍ para pasar a % Masa -> NO, NUEVO.exp ya está en %
                    x_val = numbers[0] 
                    y_val = numbers[1]
                    
                    # Detectar bandera 'M' (Move) que indica inicio de nuevo segmento
                    if 'M' in line:
                        # Si ya teníamos datos acumulados, cerramos ese segmento
                        if current_block["current_segment"]:
                            current_block["segments"].append(np.array(current_block["current_segment"]))
                            current_block["current_segment"] = []
                        
                        # Añadimos el punto inicial del nuevo segmento
                        current_block["current_segment"].append([x_val, y_val])
                    else:
                        # Continuación de línea
                        current_block["current_segment"].append([x_val, y_val])

    # Añadir el último bloque si quedó pendiente
    if current_block:
        if current_block["current_segment"]:
            current_block["segments"].append(np.array(current_block["current_segment"]))
        blocks.append(current_block)
        
    return blocks

# ==========================================
# 3. GRAFICACIÓN
# ==========================================

data_blocks = parse_thermocalc_exp(exp_content)

# Calc dimensions: Double column width, height = width (Square)
width = DOUBLE_COLUMN_MM * MM_TO_INCH
height = width
fig, ax = plt.subplots(figsize=(width, height))

# Colores personalizados para simular Thermo-Calc
colors = ['blue', 'green', 'red', 'brown', 'purple', 'orange']
color_idx = 0

print(f"Se encontraron {len(data_blocks)} bloques de datos.")

for i, block in enumerate(data_blocks):
    # Determinar etiqueta para la leyenda
    # Usamos la última fase listada ($F0) como nombre principal, o combinamos
    label_name = " / ".join(block["phases"]) if block["phases"] else f"Data {i+1}"

    # SUSPENDER (OCULTAR) lineas de IONIC_LIQ y IONIC_LIQ#2 ESPECIFICAS (L1 + L2 solas)
    # El usuario dijo "esa linea solamente, no todo". 
    # Analisis mostró bloques con ['IONIC_LIQ#1', 'IONIC_LIQ#2']. Esto es la laguna de miscibilidad pura.
    phases_set = set(block["phases"])
    
    # 1. Hide Miscibility Gap (L1+L2 only)
    if phases_set == {'IONIC_LIQ#1', 'IONIC_LIQ#2'}:
        continue

    if phases_set == {'IONIC_LIQ#1', 'IONIC_LIQ#2','HALITE#2'}:
        continue
    
    # Asignar un color por bloque
    c = colors[color_idx % len(colors)]
    color_idx += 1
    
    first_segment = True
    for segment in block["segments"]:
        if len(segment) > 0:
            # Solo ponemos etiqueta en el primer segmento para no repetir en la leyenda
            lbl = label_name if first_segment else "_nolegend_"
            
            # X = Columna 0 (FeO), Y = Columna 1 (SiO2) -> MATCH .exp file
            # .exp file says: XTEXT W(FEO), YTEXT W(SIO2)
            # User wants X=FeO, Y=SiO2. So we use columns directly: 0 for X, 1 for Y.
            ax.plot(segment[:, 0], segment[:, 1], 
                    color=c, 
                    linewidth=1.2, 
                    label=lbl)
            first_segment = False

# ==========================================
# 4. LEER DATOS ADICIONALES (EXCEL) y GRAFICAR
# ==========================================
csv_filename = "viscfraccion.xls"
csv_path = os.path.join(script_dir, csv_filename)

print(f"Intentando leer CSV desde: {csv_path}")

try:
    # Read Excel file
    df = pd.read_excel(csv_path)
    print("Excel leído exitosamente.")
    
    # Iterate over columns in steps of 3
    num_cols = len(df.columns)
    for i in range(0, num_cols, 3):
        if i+2 >= num_cols: break
        
        # Header is the label (e.g., "0.3 IONIC...")
        col_label_raw = df.columns[i]
        # Extract just the number (first part)
        label_val = col_label_raw.split()[0]
        
        # Determine color and type
        is_fraction = "Ratio" in col_label_raw
        line_color = 'red' if is_fraction else 'blue'

        # Helper para suavizado
        def smooth_curve(x_in, y_in):
            # Requiere al menos 4 puntos para k=3
            if len(x_in) < 4:
                return x_in, y_in
            
            # splprep no le gustan puntos duplicados consecutivos
            # Crear dataframe para filtrar rápido
            tmp_df = pd.DataFrame({'x': x_in, 'y': y_in})
            tmp_df = tmp_df.drop_duplicates()
            if len(tmp_df) < 4:
                return x_in, y_in

            try:
                # s=0 fuerza a pasar por todos los puntos. k=3 es cúbico.
                tck, u = splprep([tmp_df['x'], tmp_df['y']], s=0.0, k=3)
                # Generar mas puntos para que se vea suave
                u_new = np.linspace(u.min(), u.max(), 300)
                x_new, y_new = splev(u_new, tck)
                return x_new, y_new
            except Exception as e:
                # Si falla (ej. datos muy feos), devuelve original
                return x_in, y_in

        # CSV structure: i (Label), i+1 (SiO2), i+2 (FeO)
        # User wants X=FeO, Y=SiO2.
        # So we take i+2 for X, i+1 for Y.
        subset = df.iloc[:, [i+2, i+1]]
        subset.columns = ['x', 'y']

        # =========================================
        # LOGIC FOR RED LINES (Fraction): SPLIT & MULTI-LABEL
        # =========================================
        if is_fraction:
            valid_chunks = []
            current_chunk = []
            prev_row = None
            
            for _, row in subset.iterrows():
                # Check 1: Is it NaN? (Gap in data)
                if row.isna().any():
                    if current_chunk:
                        valid_chunks.append(pd.DataFrame(current_chunk))
                        current_chunk = []
                    prev_row = None
                    continue
                
                # Check 2: Euclidean Distance (Jump > 5% composition)
                if prev_row is not None:
                    dist = np.sqrt((row['x'] - prev_row['x'])**2 + (row['y'] - prev_row['y'])**2)
                    if dist > 5.0:
                        if current_chunk:
                            valid_chunks.append(pd.DataFrame(current_chunk))
                            current_chunk = []
                
                current_chunk.append(row)
                prev_row = row
            
            if current_chunk:
                valid_chunks.append(pd.DataFrame(current_chunk))
            
            # Graficar Y Etiquetar CADA segmento significativo
            for chunk in valid_chunks:
                if chunk.empty: continue
                
                # Graficar segmento SUAVIZADO
                sx, sy = smooth_curve(chunk['x'].values, chunk['y'].values)
                ax.plot(sx, sy, 
                        linestyle='-.', 
                        linewidth=0.8, 
                        color=line_color,
                        label='_nolegend_')
                
                # Etiquetar si el segmento tiene suficientes puntos (>3)
                if len(chunk) > 3:
                    lbl_chunk = chunk
                    mid_idx = len(lbl_chunk) // 2
                    lbl_x = lbl_chunk.iloc[mid_idx]['x']
                    lbl_y = lbl_chunk.iloc[mid_idx]['y']
                    
                    try:
                        val_float = float(label_val)
                        label_str = str(round(val_float, 2))
                    except:
                        label_str = label_val
                    
                    ax.text(lbl_x, lbl_y, label_str, 
                            fontsize=10, 
                            color=line_color, 
                            weight='bold',
                            verticalalignment='center', 
                            horizontalalignment='center',
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))

        # =========================================
        # LOGIC FOR BLUE LINES (Viscosity): RAW DATA WITH GAP SPLITTING
        # =========================================
        else:
            subset = subset.dropna()
            if len(subset) == 0: continue
            
            # 1. NO Ordenar por X (respetar orden del archivo para curvas complejas)
            # subset = subset.sort_values(by='x')
            
            # 2. Detectar SALTOS para dividir en líneas (igual que en Fracción)
            valid_chunks = []
            current_chunk = []
            prev_row = None
            
            for _, row in subset.iterrows():
                # Check Euclidean Distance (Jump > 5.0 composition units)
                if prev_row is not None:
                    dist = np.sqrt((row['x'] - prev_row['x'])**2 + (row['y'] - prev_row['y'])**2)
                    if dist > 5.0:
                        if current_chunk:
                            valid_chunks.append(pd.DataFrame(current_chunk))
                            current_chunk = []
                
                current_chunk.append(row)
                prev_row = row
            
            if current_chunk:
                valid_chunks.append(pd.DataFrame(current_chunk))
            
            # 3. Graficar CADA segmento
            for chunk in valid_chunks:
                if chunk.empty: continue
                
                # Graficar segmento SUAVIZADO
                sx, sy = smooth_curve(chunk['x'].values, chunk['y'].values)
                ax.plot(sx, sy, 
                        linestyle='-.', 
                        linewidth=0.8, 
                        color=line_color,
                        label='_nolegend_')
                
                # Etiqueta en el centro de CADA segmento
                if len(chunk) > 0: # Label even small chunks if needed, or set > 1
                    mid_idx = len(chunk) // 2
                    lbl_x = chunk.iloc[mid_idx]['x']
                    lbl_y = chunk.iloc[mid_idx]['y']
                    
                    try:
                        val_float = float(label_val)
                        label_str = str(round(val_float, 2))
                    except:
                        label_str = label_val
                        
                    ax.text(lbl_x, lbl_y, label_str, 
                            fontsize=10, 
                            color=line_color, 
                            weight='bold',
                            verticalalignment='center', 
                            horizontalalignment='center',
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))
        
except FileNotFoundError:
    print(f"No se encontró el archivo Excel: {csv_filename}")
except Exception as e:
    print(f"Error procesando Excel: {e}")

# ==========================================
# 5. CONFIGURACIÓN DEL DIAGRAMA (Estilo Imagen)
# ==========================================

# Título y Etiquetas
ax.set_title("Diagrama FeO-SiO$_2$", fontsize=14)
ax.set_xlabel("Mass percent FeO", fontsize=12)
ax.set_ylabel("Mass percent SiO$_2$", fontsize=12)

# Límites
ax.set_xlim(0, 34)
ax.set_ylim(0, 34)

# Grid y Leyenda
ax.grid(True, linestyle='--', alpha=0.6)
# Leyenda más pequeña, sin marco ni fondo
# leg = ax.legend(loc='upper right', fontsize=7, frameon=False)
# leg.set_draggable(True)

# Leyenda personalizada visual
custom_lines = [
    Line2D([0], [0], color='blue', lw=0.8, linestyle='-.'),
    Line2D([0], [0], color='red', lw=0.8, linestyle='-.')
]
ax.legend(custom_lines, ['Viscosidad (mPa*s)', 'Fracción de liquido'], 
          loc='upper left', fontsize=10, frameon=True, 
          facecolor='white', framealpha=0.9, edgecolor='lightgray')

# Re-agregar la primera leyenda
# ax.add_artist(leg)



# (Opcional) Dibujar la línea de contorno triangular si quisieras mostrar el límite
# x + y <= 100 (aunque con zoom a 50 no se ve el borde CaO puro)

plt.tight_layout()
output_path = os.path.join(figures_dir, 'diagrama_output.png')
plt.savefig(output_path, bbox_inches='tight')
print(f"Guardado: {output_path}")
# plt.show()

# ==========================================
# 6. GENERACIÓN DE DIAGRAMA PURO (FASE SOLA con ETIQUETAS)
# ==========================================
def generate_pure_phase_diagram():
    print("\n--- Generando Diagrama de Fases Puro ---")
    
    # 1. Leer archivo diagrama fases.exp
    pure_exp_filename = "diagrama fases.exp"
    pure_exp_path = os.path.join(script_dir, pure_exp_filename)
    
    try:
        with open(pure_exp_path, 'r', encoding='latin-1') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: No se encontró {pure_exp_filename} para diagrama puro.")
        return

    blocks_pure = parse_thermocalc_exp(content)
    print(f"Bloques encontrados en diagrama puro: {len(blocks_pure)}")

    # 2. Setup Figure (Mismo tamaño cuadrado y estilo)
    width = DOUBLE_COLUMN_MM * MM_TO_INCH
    height = width
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Simular Colores
    colors_pure = ['black', 'blue', 'green', 'purple', 'brown', 'orange', 'gray', 'olive', 'cyan']
    
    # Diccionario para evitar etiquetas encimadas excesivas
    # Key: LabelStr, Value: list of (x,y) positions
    placed_labels = []

    for i, block in enumerate(blocks_pure):
        # Filtros iguales al principal
        phases_set = set(block["phases"])
        if phases_set == {'IONIC_LIQ#1', 'IONIC_LIQ#2'}: continue
        if phases_set == {'IONIC_LIQ#1', 'IONIC_LIQ#2','HALITE#2'}: continue
        
        # Nombre bonito para etiqueta
        # Limpiar caracteres raros si los hay
        raw_phases = [p.replace('_', ' ').replace('#1', '').replace('#2', '') for p in block["phases"]]
        label_text = " + ".join(raw_phases)
        
        # Color consistente por conjunto de fases? 
        # Hash del set de fases para color fijo
        color_seed = sum(map(ord, "".join(sorted(block["phases"])))) 
        c = colors_pure[color_seed % len(colors_pure)]
        
        for segment in block["segments"]:
            if len(segment) == 0: continue
            
            # SCALE DATA: diagrama fases.exp is in fraction (0-1), layout is 0-34
            seg_scaled = segment * 100.0
            
            # Plot Line
            ax.plot(seg_scaled[:, 0], seg_scaled[:, 1], color=c, linewidth=0.8) 
            # (Usamos negro para lineas de bordes en diagrama puro, o color? User didn't specify, standard is black lines)

    # --- LABELS MANUALES (Generado por script, ajustar coordenadas aqui) ---
    # Para ROTAR: cambiar rotation=0 por el angulo deseado (ej. rotation=45)
    ax.text(5.89, 13.16, 'Liquido + (Mn,Fe)O + (Fe,Ca)O + 3CS', rotation=0, fontsize=8, color='brown', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
    ax.text(10, 15.4, 'Liquido + (Fe,Ca)O + 3CS', rotation=-25, fontsize=4, color='green', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
    ax.text(5.88, 18.70, 'Liquido + C₂S + (Fe,Ca),O + 3CS', rotation=-28, fontsize=8, color='brown', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
    ax.text(8, 21.5, 'Liquido + C₂S + (Fe,Ca)O', rotation=-32, fontsize=8, color='blue', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
    ax.text(8.21, 26, 'Liquido + C₂S', rotation=-30, fontsize=8, color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
    ax.text(24.41, 10, 'Liquido + (Fe,Ca)O', rotation=-30, fontsize=8, color='blue', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
    ax.text(15.64, 5.87, 'Liquido + (Fe,Ca)O + (Mn,Fe)O', rotation=0, fontsize=8, color='blue', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))
    ax.text(25,20, 'Liquido', rotation=0, fontsize=12, color='blue', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))

    # --- PUNTOS EAF LITERATURA ---
    eaf_points_all = puntos_EAF_completos()
    
    # 1. Identificar Puntos en rango de gráfico (0-34)
    valid_points = [p for p in eaf_points_all if 0 <= p['FeO'] <= 34 and 0 <= p['SiO2'] <= 34]
    
    # 2. Filtrar Referencia vs Resto
    # Usamos la funcion de filtro sobre el total
    ref_candidates = filtrar_referencia_5_5_7(eaf_points_all)
    
    points_ref = []     # Cumplen criterio (ROJO)
    points_other = []   # No cumplen criterio (AZUL)
    
    for p in valid_points:
        if p in ref_candidates:
            points_ref.append(p)
        else:
            points_other.append(p)

    # 3. Graficar "OTROS" (AZUL + LABEL)
    if points_other:
        print(f"Graficando {len(points_other)} puntos fuera de ref (Azul)...")
        other_x = [p['FeO'] for p in points_other]
        other_y = [p['SiO2'] for p in points_other]
        
        ax.scatter(other_x, other_y, c='blue', edgecolors='black', linewidth=0.5, marker='o', s=15, zorder=5, label='Otros EAF')
        
        for p in points_other:
            ax.text(p['FeO'] - 0.2, p['SiO2'], p['label'], 
                    fontsize=5, color='darkblue', 
                    ha='right', va='center', 
                    rotation=0,
                    zorder=6)

    # 4. Graficar "REFERENCIA" (ROJO + LABEL)
    if points_ref:
        print(f"Graficando {len(points_ref)} puntos de referencia (Rojo)...")
        ref_x = [p['FeO'] for p in points_ref]
        ref_y = [p['SiO2'] for p in points_ref]
        
        ax.scatter(ref_x, ref_y, c='red', edgecolors='black', linewidth=0.5, marker='o', s=20, zorder=6, label='EAF Ref (5%Al/7%Mg)')
        
        for p in points_ref:
            ax.text(p['FeO'] - 0.2, p['SiO2'], p['label'], 
                    fontsize=5, color='darkred', 
                    ha='right', va='center', 
                    rotation=0,
                    zorder=7)
        
    # --- LINEAS DE BASICIDAD (CaO/SiO2) ---
    # Formula dada: (68-%FeO-%SiO2)+15 = %CaO
    # => %CaO + %FeO + %SiO2 = 83
    # Ratio B = %CaO / %SiO2
    # Sustituyendo CaO = B * SiO2:
    # B * SiO2 + SiO2 = 83 - FeO
    # SiO2 (B + 1) = 83 - FeO
    # SiO2 = (83 - FeO) / (B + 1)
    
    basicities = [1.0, 2.0, 2.5, 3.0]
    x_fe_range = np.linspace(0, 34, 100) # Rango X del grafico
    
    print("Graficando lineas de basicidad (1, 2, 2.5, 3)...")
    
    for B in basicities:
        # Calcular Y (SiO2)
        y_si_vals = (83 - x_fe_range) / (B + 1)
        
        # Filtrar para plotear solo dentro del rango visible Y (0-34)
        mask = (y_si_vals >= 0) & (y_si_vals <= 34)
        if not np.any(mask): 
            continue
            
        x_plot = x_fe_range[mask]
        y_plot = y_si_vals[mask]
        
        # Plotear linea (Mas oscura)
        ax.plot(x_plot, y_plot, color='black', linestyle='--', linewidth=0.8, alpha=0.7, zorder=2)
        
        # Etiqueta
        # Calculamos rotación aproximada
        # dy/dx = -1 / (B+1). En un plot cuadrado con escalas iguales, el angulo es arctan(dy/dx)
        angle_rad = np.arctan(-1 / (B + 1))
        angle_deg = np.degrees(angle_rad)
        
        # Posición del label: Aproximadamente a la mitad de la linea visible o en un extremo
        # Por defecto al principio (Arriba)
        lbl_idx = 0 
        
        # AJUSTE MANUAL: B=1 mas abajo
        if B == 1.0:
            lbl_idx = int(len(x_plot) * 0.4) # 40% del camino abajo
        
        lbl_x = x_plot[lbl_idx]
        lbl_y = y_plot[lbl_idx]
        
        # Pequeño ajuste para que no pise la linea
        ax.text(lbl_x + 0.5, lbl_y + 0.5, f"B={B}", 
                fontsize=7, color='black', 
                ha='left', va='bottom', 
                rotation=angle_deg,
                zorder=3)

    # Configuración Ejes
    ax.set_title("Diagrama de Fases Puro", fontsize=14)
    ax.set_xlabel("Mass percent FeO", fontsize=12)
    ax.set_ylabel("Mass percent SiO$_2$", fontsize=12)
    ax.set_xlim(0, 34)
    ax.set_ylim(0, 34)
    ax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.tight_layout()
    output_pure_path = os.path.join(figures_dir, 'diagrama_fases_puro.png')
    plt.savefig(output_pure_path, bbox_inches='tight')
    print(f"Guardado: {output_pure_path}")
    # plt.show()

# EJECUTAR FUNCION
generate_pure_phase_diagram()