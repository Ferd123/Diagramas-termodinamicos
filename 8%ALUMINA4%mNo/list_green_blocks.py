import os
import re

ALLOWED_LIQUIDS = {"IONIC_LIQ#1", "IONIC_LIQ#2", "IONIC_LIQ#3"}
ALLOWED_PHASES = ALLOWED_LIQUIDS | {"HALITE#1", "HALITE#2"}

def format_phase_label(phases):
    seen = []
    for p in phases:
        if p not in seen:
            seen.append(p)
    replacements = {
        "IONIC_LIQ#1": "Liquido#1",
        "IONIC_LIQ#2": "Liquido#2",
        "IONIC_LIQ#3": "Liquido#3",
        "HALITE#1": "Halite#1",
        "HALITE#2": "Halite#2",
    }
    pretty = [replacements.get(p, p.replace("_", " ")) for p in seen]
    return " + ".join(pretty)

def parse_blocks(content):
    blocks = []
    current = None
    block_id_pattern = re.compile(r"\$\s*BLOCK\s*#(\d+)", re.IGNORECASE)

    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("$ BLOCK") or line.startswith("$BLOCK"):
            if current:
                blocks.append(current)
            block_id = None
            match = block_id_pattern.search(line)
            if match:
                block_id = int(match.group(1))
            current = {"block_id": block_id, "phases": []}
        elif line.startswith("$F0") or line.startswith("$E"):
            if current is not None:
                phase_name = line.split(maxsplit=1)[1] if len(line.split()) > 1 else "Unknown"
                current["phases"].append(phase_name)
        elif line.startswith("BLOCKEND"):
            if current:
                blocks.append(current)
                current = None

    if current:
        blocks.append(current)
    return blocks

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_path = os.path.join(script_dir, "40.exp")

    with open(exp_path, "r", encoding="latin-1") as f:
        content = f.read()

    blocks = parse_blocks(content)

    print("Blocks in 40.exp (green):")
    for b in blocks:
        phases_set = set(b["phases"])
        has_halite = "HALITE#1" in phases_set or "HALITE#2" in phases_set
        only_liquid_halite = phases_set.issubset(ALLOWED_PHASES)
        label = format_phase_label(b["phases"])
        flag = "PLOT" if has_halite and only_liquid_halite else "SKIP"
        print(f"#{b['block_id']}: {label} [{flag}]")

if __name__ == "__main__":
    main()
