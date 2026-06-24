"""
Inject new table rows (with CNSG as 95% CI, ^\ast flag) into both LaTeX files,
and replace all \dagger with \ast throughout.
"""
import re
import subprocess
import os

BASE = r"C:\Users\montr\OneDrive - University of Southampton\wind_pv_synergy"

# ---------- 1. Generate rows ----------
result = subprocess.run(
    ["python", os.path.join(BASE, "code", "generate_tables_with_cnsg.py")],
    capture_output=True, text=True, cwd=BASE
)
raw = result.stdout

sections = {}
current = None
for line in raw.splitlines():
    m = re.match(r"% ---- Table: (\w+) eps=([\d.]+) ----", line)
    if m:
        current = f"{m.group(1)}_{m.group(2)}"
        sections[current] = []
    elif current and line.startswith("    "):
        sections[current].append(line)

def new_body(key):
    return "\n".join(sections[key])


# ---------- 2. Replace midrule..bottomrule for a given label ----------
def replace_body(tex, label, rows):
    marker = "\\label{" + label + "}"
    idx = tex.find(marker)
    if idx < 0:
        raise ValueError(f"Label not found: {label}")
    after = tex[idx + len(marker):]
    mid_pos = after.find("\\midrule")
    if mid_pos < 0:
        raise ValueError(f"\\midrule not found after {label}")
    bot_pos = after.find("\\bottomrule")
    if bot_pos < 0:
        raise ValueError(f"\\bottomrule not found after {label}")
    mid_end = mid_pos + len("\\midrule")
    new_after = after[:mid_end] + "\n" + rows + "\n    " + after[bot_pos:]
    return tex[:idx + len(marker)] + new_after


# ---------- 3. Fix \dagger -> \ast ----------
def fix_dagger(tex):
    return tex.replace("\\dagger", "\\ast")


# ---------- 4. Update section_6_absolute_profits.tex ----------
path6 = os.path.join(BASE, "writing", "section_6_absolute_profits.tex")
with open(path6, encoding="utf-8") as f:
    tex6 = f.read()

tex6 = fix_dagger(tex6)
tex6 = replace_body(tex6, "tab:naive_e01", new_body("Naive_0.01"))
tex6 = replace_body(tex6, "tab:naive_e05", new_body("Naive_0.05"))
tex6 = replace_body(tex6, "tab:naive_e10", new_body("Naive_0.1"))

with open(path6, "w", encoding="utf-8") as f:
    f.write(tex6)
print("section_6_absolute_profits.tex updated")


# ---------- 5. Update section_5_rolling_window.tex ----------
path5 = os.path.join(BASE, "writing", "section_5_rolling_window.tex")
with open(path5, encoding="utf-8") as f:
    tex5 = f.read()

tex5 = fix_dagger(tex5)
tex5 = replace_body(tex5, "tab:rolling_e01", new_body("Battery_0.01"))
tex5 = replace_body(tex5, "tab:rolling_e05", new_body("Battery_0.05"))
tex5 = replace_body(tex5, "tab:rolling_e10", new_body("Battery_0.1"))

with open(path5, "w", encoding="utf-8") as f:
    f.write(tex5)
print("section_5_rolling_window.tex updated")
