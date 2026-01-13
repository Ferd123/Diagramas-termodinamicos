import re
import numpy as np
import os

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
        else:
            if current_block and not (line.startswith("BLOCK") or line.startswith("$")):
                numbers = [float(x) for x in number_pattern.findall(line)]
                if len(numbers) >= 2:
                    current_block["current_segment"].append([numbers[0], numbers[1]])
    if current_block: blocks.append(current_block)
    return blocks

script_dir = r"c:\Users\ELANOR\Documents\DIAGRAMA FEO-CAO-SIO2-MGO-AL2O3-MNO"
path = os.path.join(script_dir, "diagrama fases.exp")

with open(path, 'r', encoding='latin-1') as f:
    content = f.read()

blocks = parse_thermocalc_exp(content)
unique_labels = {}

for block in blocks:
    phases_set = set(block["phases"])
    if phases_set == {'IONIC_LIQ#1', 'IONIC_LIQ#2'}: continue
    if phases_set == {'IONIC_LIQ#1', 'IONIC_LIQ#2','HALITE#2'}: continue
    
    raw_phases = [p.replace('_', ' ').replace('#1', '').replace('#2', '') for p in block["phases"]]
    label_text = " + ".join(raw_phases)
    
    for segment in block["segments"]:
        if len(segment) == 0: continue
        seg_scaled = segment * 100.0
        dist = np.sqrt( (seg_scaled[-1,0]-seg_scaled[0,0])**2 + (seg_scaled[-1,1]-seg_scaled[0,1])**2 )
        
        if dist > 1.0:
            mid_idx = len(seg_scaled) // 2
            lx = seg_scaled[mid_idx, 0]
            ly = seg_scaled[mid_idx, 1]
            
            # Keep the one with the largest segment if duplicate? 
            # Or just keep the first/last?
            # Let's store all and print distinctive ones or just dedupe by text
            if label_text not in unique_labels:
                unique_labels[label_text] = (lx, ly, block["phases"])
            else:
                # Update if this segment is 'better' (e.g. valid)? 
                # For now just keep first found
                pass

colors_pure = ['black', 'blue', 'green', 'purple', 'brown', 'orange', 'gray', 'olive', 'cyan']

print("# --- LABELS MANUALES ---")
for lbl, (lx, ly, phases) in unique_labels.items():
    color_seed = sum(map(ord, "".join(sorted(phases)))) 
    c = colors_pure[color_seed % len(colors_pure)]
    print(f"    ax.text({lx:.2f}, {ly:.2f}, '{lbl}', fontsize=6, color='{c}', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2))")
