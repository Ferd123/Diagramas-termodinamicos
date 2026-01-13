import re

content = open("NUEVO.exp", 'r', encoding='latin-1').read()
blocks = []
current_block = None

lines = content.split('\n')
for line in lines:
    line = line.strip()
    if not line: continue
    if line.startswith("$ BLOCK") or line.startswith("$BLOCK"):
        if current_block: blocks.append(current_block)
        current_block = {"phases": []}
    elif line.startswith("$F0") or line.startswith("$E"):
        if current_block:
            phase_name = line.split(maxsplit=1)[1] if len(line.split()) > 1 else "Unknown"
            current_block["phases"].append(phase_name)
    elif line.startswith("BLOCKEND"):
        if current_block: blocks.append(current_block)
        current_block = None

for i, b in enumerate(blocks):
    print(f"Block {i}: {b['phases']}")
