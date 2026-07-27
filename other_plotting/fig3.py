import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
df = pd.DataFrame({
    "Month_hu": [
        "Január", "Február", "Március", "Április", "Május", "Június",
        "Július", "Augusztus", "Szeptember", "Október", "November", "December"
    ],
    "Early": [0.000, 0.078, 0.664, 0.201, 0.083, 0.000, 0.015, 0.000, 0.660, 0.112, 0.174, 0.000],
    "Late":  [0.000, 0.328, 0.472, 0.302, 0.141, 0.029, 0.000, 0.029, 0.699, 0.230, 0.172, 0.000],
    "Average_change": [0.000, 0.250, -0.193, 0.101, 0.058, 0.029, -0.015, 0.029, 0.039, 0.118, -0.002, 0.000],
    "Increasing_sites": [0, 9, 0, 6, 6, 2, 0, 2, 6, 4, 3, 0],
    "Decreasing_sites": [0, 0, 10, 3, 3, 0, 1, 0, 4, 2, 4, 0],
    "Unchanged_sites":  [10, 1, 0, 1, 1, 8, 9, 8, 0, 4, 3, 10]
})

# English month labels for the figure
df["Month"] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Delta H
df["DeltaH"] = df["Late"] - df["Early"]

x = np.arange(len(df))
width = 0.38

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [2.2, 1.4]}
)

# -------------------------
# Top panel: grouped bars
# -------------------------
ax1.bar(x - width/2, df["Early"], width, label="Early period (1961–1990)")
ax1.bar(x + width/2, df["Late"],  width, label="Late period (1995–2024)")

ax1.set_ylabel("Shannon entropy")
ax1.set_title(
    "Monthly changes in Shannon entropy between 1961–1990 and 1995–2024",
    fontweight="bold",
    pad=12
)
ax1.legend(frameon=False, loc="upper right")
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# -------------------------
# Bottom panel: difference bars
# -------------------------
colors = ["tab:blue" if v >= 0 else "tab:red" for v in df["DeltaH"]]
bars = ax2.bar(x, df["DeltaH"], color=colors)

ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_ylabel("ΔH (late − early)")
ax2.set_xticks(x)
ax2.set_xticklabels(df["Month"])
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Value labels on difference bars
for i, v in enumerate(df["DeltaH"]):
    if v >= 0:
        ax2.text(i, v + 0.01, f"{v:+.3f}", ha="center", va="bottom", fontsize=9)
    else:
        ax2.text(i, v - 0.01, f"{v:+.3f}", ha="center", va="top", fontsize=9)

# Optional: annotate number of increasing / decreasing / unchanged sites
ymin = min(df["DeltaH"].min() - 0.08, -0.28)
ymax = max(df["DeltaH"].max() + 0.08, 0.28)
ax2.set_ylim(ymin, ymax)

for i, row in df.iterrows():
    ax2.text(
        i, ymin + 0.02,
        f"↑{row['Increasing_sites']}  ↓{row['Decreasing_sites']}  ={row['Unchanged_sites']}",
        ha="center", va="bottom", fontsize=8
    )

plt.tight_layout()
plt.savefig("figure3_monthly_shannon_entropy.png", dpi=300, bbox_inches="tight")
plt.show()