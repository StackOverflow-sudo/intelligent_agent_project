# patch_group4.py
from pathlib import Path
import re

src = Path("group4.py").read_text(encoding="utf-8")

# 1) COST_SAFETY
src = re.sub(r"COST_SAFETY\s*=\s*0\.25", "COST_SAFETY = 0.08", src)

# 2) choose_bid_risk_aware multipliers
src = re.sub(
    r"for m in \[[^\]]+\]:\s*\n\s*candidates\.add\(c_eff\s*\*\s*m\)",
    "for m in [0.70, 0.80, 0.90, 1.00, 1.05, 1.10, 1.20, 1.35, 1.50, 2.00, 3.00]:\n            candidates.add(c_eff * m)",
    src,
    flags=re.MULTILINE,
)

# 3) remove hard floor block inside choose_bid_risk_aware
src = re.sub(
    r"floor\s*=\s*c_eff\s*\*\s*\(1\.0\s*\+\s*self\.COST_SAFETY\)\s*\n\s*floor\s*\*=\s*1\.10\s*\n\s*candidates\s*=\s*sorted\(b\s*for\s*b\s*in\s*candidates\s*if\s*b\s*>=\s*floor\)",
    "candidates = sorted(b for b in candidates if b >= 0.60 * c_eff)",
    src,
    flags=re.MULTILINE,
)

# 4) safety_floor & min_margin
src = re.sub(r"safety_floor\s*=\s*0\.10", "safety_floor = 0.03", src)
src = re.sub(r"min_margin\s*=\s*0\.08", "min_margin = 0.02", src)

# 5) inject lose-streak relax right after min_margin line (first occurrence)
relax_block = """
        lose_streak = getattr(self, "_no_win_rounds", 0)
        if lose_streak > 0:
            safety_floor = max(0.0, safety_floor - 0.01 * min(lose_streak, 5))
            min_margin  = max(0.0, min_margin  - 0.01 * min(lose_streak, 5))
""".rstrip("\n")

src = re.sub(
    r"(min_margin\s*=\s*0\.02[^\n]*\n)",
    r"\1" + relax_block + "\n",
    src,
    count=1,
)

# 6) inject no-win counter after RECEIVE log (first occurrence)
counter_block = """
        if not hasattr(self, "_no_win_rounds"):
            self._no_win_rounds = 0
        if contracts and len(contracts) > 0:
            self._no_win_rounds = 0
        else:
            self._no_win_rounds += 1
""".rstrip("\n")

src = re.sub(
    r"(force_log\(f\"\[RECEIVE\][^\n]*\n)",
    r"\1" + counter_block + "\n",
    src,
    count=1,
)

Path("group4_patched.py").write_text(src, encoding="utf-8")
print("Wrote group4_patched.py")
