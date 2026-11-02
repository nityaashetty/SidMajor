"""Re-render the paper's data figures in Arial Narrow (to match the IJCRSEE body text) as PNGs for the .docx.

Reuses Paper/analysis/figures.py unchanged; only the font, the output folder and the file format differ,
so the paper's own PDFs in Paper/figures/ are not touched.

    python docx_build/make_figures.py      (run from Report_IJCRSEE/)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.figure import Figure

BUILD = Path(__file__).resolve().parent
ANALYSIS = BUILD.parent.parent / "Paper" / "analysis"
sys.path.insert(0, str(ANALYSIS))
import figures as F  # noqa: E402

for f in ("ARIALN.TTF", "ARIALNB.TTF", "ARIALNI.TTF"):
    if (Path("C:/Windows/Fonts") / f).exists():
        font_manager.fontManager.addfont(str(Path("C:/Windows/Fonts") / f))
plt.rcParams.update({"font.family": "Arial", "font.stretch": "condensed", "mathtext.fontset": "custom",
                     "mathtext.rm": "Arial", "font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})

_orig = Figure.savefig


def _png(self, fname, *a, **k):
    fname = Path(fname)
    return _orig(self, BUILD / (fname.stem + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)


Figure.savefig = _png
F.FIG = BUILD
F.alignment(); F.protocol(); F.roc_internal_external(); F.confusion()
print("figures written to", BUILD)
