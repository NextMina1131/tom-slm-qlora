"""Regenerate Figure 1 (summary of performance changes, seed 42).

The values are taken from the manuscript's tables (overall results with their
confidence intervals and McNemar significance for Panel A, the category-level
table for Panel B, the ability-level table for Panel C), so no inference is
needed. Panel B uses the six categories after merging the two spellings of
"Non-Literal Communication" in the source data.

Output: Figure1.png in the current directory, 1000 dpi. The figure is drawn at its
printed width (7.0 in), so the font sizes below are the printed sizes; Panels A and B
sit side by side above Panel C.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8})

OUT_PATH = "Figure1.png"

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
panelC_sig = [True, True, True, True, True, True, False, False, True, True, False]

fig = plt.figure(figsize=(7.0, 7.4), layout="constrained")
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.45], width_ratios=[1.0, 1.0])
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]

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
ax.set_xlabel(r"$\Delta$ Accuracy (points)")
ax.set_title("(A) Change by benchmark\n(seed 42, with 95% CI)", loc="left")
fig.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, color=SIG_COLOR, label="Holm-corrected significant"),
    plt.Rectangle((0, 0), 1, 1, color=NOTSIG_COLOR, label="Not significant"),
], loc="outside upper center", ncols=2, frameon=False)

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
ax.set_xlabel(r"$\Delta$ Accuracy (points)")
ax.set_title("(B) ToMBench, by category\n(in-domain, with 95% CI)", loc="left")

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
                ha="left" if d >= 0 else "right", fontsize=10, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(panelC_labels)
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel(r"$\Delta$ Accuracy (percentage points)")
ax.set_title("(C) Top ToMBench ability-level improvements, ordered by N (largest first);\n"
             "* marks bootstrap 95% CI excluding zero", loc="left")

fig.get_layout_engine().set(h_pad=0.08, w_pad=0.08, hspace=0.04, wspace=0.06)
fig.savefig(OUT_PATH, dpi=1000)
print(f"Saved {OUT_PATH}")

import PIL.Image
im = PIL.Image.open(OUT_PATH)
print(f"Size: {im.size}, DPI: {im.info.get('dpi')}")
