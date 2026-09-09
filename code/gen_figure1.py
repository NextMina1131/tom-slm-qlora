"""Regenerate Figure 1 (summary of performance changes, seed 42, corrected pipeline).

Rewritten from scratch: the original gen_figure1.py is not present in this
repository. Source data is copied by hand from the corresponding LaTeX tables
in tom-slm-qlora_MDPI_source_0909/main.tex (Table tab:overall + Table6 CIs +
Table tab:mcnemar significance for Panel A; Table tab7 for Panel B; Table tab8
for Panel C) so that regenerating this figure never requires re-running
inference. Panel B categories are taken directly from Table 7's already
deduplicated 6-category breakdown, which avoids the capitalization-duplicate
bug ("Non-Literal Communication" vs "Non-Literal communication") present in
the previous Figure1.png.

Output: 300 dpi, vertical stacked layout (matches the 0828 layout, which had
higher resolution and more legible Panel C labels than the 0907 horizontal
layout).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = "/mnt/e/파인튜닝/ToMBench-main/tom-slm-qlora_MDPI_source_0909/Figure1.png"

SIG_COLOR = "#3a6ea5"
NOTSIG_COLOR = "#b0b0b0"
POS_COLOR = "#3a8a5a"
NEG_COLOR = "#b5533c"

# ---- Panel A: cross-benchmark delta accuracy, seed 42 (Table tab:overall / Table6 / tab:mcnemar) ----
panelA_labels = ["ToMBench\n(in-domain)", "ToMi", "OpenToM", "SocialIQa", "Hi-ToM"]
panelA_delta = [13.66, 6.70, -3.90, -6.00, -0.10]
panelA_ci_lo = [10.05, 3.49, -6.26, -8.89, -3.17]
panelA_ci_hi = [17.23, 9.85, -1.60, -3.10, 2.90]
panelA_sig = [True, True, True, True, False]

# ---- Panel B: ToMBench category-level delta (Table tab7) ----
panelB_labels = ["Knowledge", "Belief", "Non-Literal\nCommunication", "Intention", "Desire", "Emotion"]
panelB_delta = [29.41, 17.74, 14.41, 12.24, 0.00, -0.81]
panelB_ci_lo = [17.24, 11.93, 6.54, 2.04, -14.58, -9.24]
panelB_ci_hi = [41.67, 23.70, 22.07, 22.45, 14.29, 7.14]
panelB_sig = [True, True, True, True, False, False]

# ---- Panel C: top ToMBench ability-level improvements (Table tab8), ordered by N descending ----
panelC_labels = [
    "Faux pas (N=168)",
    "Content false beliefs (N=60)",
    "Information–knowledge links (N=60)",
    "Location false beliefs +\nsecond-order beliefs (N=30)",
    "Sequence false beliefs (N=30)",
    "Hidden emotions (N=24)",
    "Egocentric lies (N=12)",
    "Involuntary lies (N=12)",
    "Knowledge–pretend play\nlinks (N=9)",
    "Irony/sarcasm (N=6)",
    "Prediction of actions (N=4)",
]
panelC_delta = [13.69, 31.67, 30.00, 26.67, 23.33, 16.67, 25.00, 16.67, 88.89, 50.00, 50.00]
panelC_sig = [True, True, True, True, True, True, False, True, True, True, False]

fig, axes = plt.subplots(3, 1, figsize=(7.5, 13.5), dpi=300,
                          gridspec_kw={"height_ratios": [1.0, 0.85, 1.35]})

# Panel A
ax = axes[0]
y = np.arange(len(panelA_labels))
colors = [SIG_COLOR if s else NOTSIG_COLOR for s in panelA_sig]
err_lo = [d - lo for d, lo in zip(panelA_delta, panelA_ci_lo)]
err_hi = [hi - d for d, hi in zip(panelA_delta, panelA_ci_hi)]
ax.barh(y, panelA_delta, color=colors, height=0.6, xerr=[err_lo, err_hi],
        capsize=3, ecolor="black", error_kw={"elinewidth": 1})
ax.set_yticks(y)
ax.set_yticklabels(panelA_labels)
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel(r"$\Delta$ Accuracy (percentage points)")
ax.set_title("(A) In-domain and cross-benchmark accuracy change (seed 42, with 95% CI)", loc="left", fontsize=10)
ax.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, color=SIG_COLOR, label="Holm-corrected significant"),
    plt.Rectangle((0, 0), 1, 1, color=NOTSIG_COLOR, label="Not significant"),
], loc="lower right", fontsize=8, frameon=False)

# Panel B
ax = axes[1]
y = np.arange(len(panelB_labels))
colors = [SIG_COLOR if s else NOTSIG_COLOR for s in panelB_sig]
err_lo = [d - lo for d, lo in zip(panelB_delta, panelB_ci_lo)]
err_hi = [hi - d for d, hi in zip(panelB_delta, panelB_ci_hi)]
ax.barh(y, panelB_delta, color=colors, height=0.6, xerr=[err_lo, err_hi],
        capsize=3, ecolor="black", error_kw={"elinewidth": 1})
ax.set_yticks(y)
ax.set_yticklabels(panelB_labels)
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel(r"$\Delta$ Accuracy (percentage points)")
ax.set_title("(B) ToMBench category-level accuracy change (in-domain, with 95% CI)", loc="left", fontsize=10)

# Panel C
ax = axes[2]
y = np.arange(len(panelC_labels))
colors = [POS_COLOR if d >= 0 else NEG_COLOR for d in panelC_delta]
alphas = [1.0 if s else 0.45 for s in panelC_sig]
bars = ax.barh(y, panelC_delta, height=0.6)
for b, c, a in zip(bars, colors, alphas):
    b.set_color(c)
    b.set_alpha(a)
for yi, (d, s) in enumerate(zip(panelC_delta, panelC_sig)):
    if s:
        ax.text(d + (2 if d >= 0 else -2), yi, "*", va="center",
                ha="left" if d >= 0 else "right", fontsize=11, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(panelC_labels, fontsize=8)
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel(r"$\Delta$ Accuracy (percentage points)")
ax.set_title("(C) Top ToMBench ability-level improvements, ordered by N (largest first);\n* marks bootstrap 95% CI excluding zero",
             loc="left", fontsize=10)

fig.tight_layout(h_pad=2.5)
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved {OUT_PATH}")

import PIL.Image
im = PIL.Image.open(OUT_PATH)
print(f"Size: {im.size}, DPI: {im.info.get('dpi')}")
