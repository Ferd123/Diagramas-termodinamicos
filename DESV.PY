import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib as mpl

# ==========================================
# 1. CONSTANTS & STYLE
# ==========================================
SINGLE_COLUMN_MM = 85
DOUBLE_COLUMN_MM = 170
MM_TO_INCH = 1 / 25.4

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlesize": 13,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.0,
    "xtick.direction": "inout",
    "ytick.direction": "inout",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "hatch.linewidth": 0.5 
})

# ==========================================
# 2. DATA
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

# ==========================================
# 3. ANALYSIS
# ==========================================
data = puntos_EAF_completos()
df = pd.DataFrame(data)

# Filter columns of interest
cols = ["FeO", "SiO2", "Al2O3", "MgO", "CaO", "MnO"]
df_analysis = df[cols]

# Calculate stats
mean_vals = df_analysis.mean()
std_vals = df_analysis.std()

print("--- ESTADÍSTICAS DE ELEMENTOS (EAF SLAG) ---")
print(f"{'Elemento':<10} | {'Promedio (%)':<15} | {'Desv. Std':<10}")
print("-" * 45)
for col in cols:
    print(f"{col:<10} | {mean_vals[col]:<15.4f} | {std_vals[col]:<10.4f}")
print("-" * 45)

# ==========================================
# 4. PLOTTING
# ==========================================
# Setup figure
width = DOUBLE_COLUMN_MM * MM_TO_INCH
height = width * 0.7 # Aspect ratio
fig, ax = plt.subplots(figsize=(width, height))

x_pos = np.arange(len(cols))
bar_width = 0.6

# 1. Bar Chart with Error Bars
bars = ax.bar(x_pos, mean_vals, yerr=std_vals, align='center', alpha=0.5, 
              ecolor='black', capsize=5, color=['brown', 'blue', 'green', 'purple', 'orange', 'cyan'],
              edgecolor='black', linewidth=1, width=bar_width, label='Promedio + SD')

# 2. Scatter Points (Individual Data)
# Add some jitter for visibility
np.random.seed(42)
jitter_strength = 0.1

for i, col in enumerate(cols):
    y_values = df_analysis[col]
    x_values = np.random.normal(i, jitter_strength, size=len(y_values))
    # Scatter points
    ax.scatter(x_values, y_values, alpha=0.7, s=25, color='black', zorder=3, edgecolors='white', linewidth=0.5)

    # Add text label for Mean above bar
    ax.text(i, mean_vals[col] + std_vals[col] + 1, f"{mean_vals[col]:.1f}±{std_vals[col]:.1f}%", 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Formatting
ax.set_ylabel("Composición (% en Masa)")
ax.set_xticks(x_pos)
ax.set_xticklabels([r"FeO", r"SiO$_2$", r"Al$_2$O$_3$", r"MgO", r"CaO", r"MnO"])
ax.set_title("Variación de Composición en Escorias EAF")
ax.grid(axis='y', linestyle=':', alpha=0.6)

# Add Legend manually if needed, or simple annotation
# ax.legend()

# Save
script_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(script_dir, "figures")
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

output_path = os.path.join(figures_dir, 'desviacion_elementos.png')
plt.tight_layout()
plt.savefig(output_path, bbox_inches='tight')
print(f"\nGráfica guardada en: {output_path}")

plt.show() # Optional, might not show in non-interactive environment