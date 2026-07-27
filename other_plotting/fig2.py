import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Data
df = pd.DataFrame({
    "Location": [
        "Brașov", "Deva", "Gheorgheni", "Győr", "Košice",
        "Kecskemét", "Keszthely", "Cluj-Napoca", "Oradea", "Pécs"
    ],
    "Early": [
        14986.536, 14630.467, 15160.063, 14612.471, 14556.085,
        15106.730, 14480.319, 15070.245, 14774.635, 14873.299
    ],
    "Late": [
        14305.696, 14131.356, 14554.893, 13735.177, 14113.650,
        13840.885, 13913.795, 14401.899, 14229.497, 13898.149
    ]
})

# Optional: relative change, if you want to use it later
df["Relative_change"] = (df["Late"] - df["Early"]) / df["Early"] * 100

# Keep original order, or sort if you want:
# df = df.sort_values("Early", ascending=True)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6.5))

y = range(len(df))

# Connecting lines
for i in range(len(df)):
    ax.plot(
        [df.loc[i, "Early"], df.loc[i, "Late"]],
        [i, i],
        linewidth=1.5,
        alpha=0.8
    )

# Points
ax.scatter(df["Early"], y, s=70, label="Early period", zorder=3)
ax.scatter(df["Late"], y, s=70, label="Late period", zorder=3)

# Axis formatting
ax.set_yticks(list(y))
ax.set_yticklabels(df["Location"])
ax.invert_yaxis()

ax.set_xlabel("Modularity")
ax.set_ylabel("")
ax.set_title(
    "Modularity of the time-based climate networks in the early and late periods.",
    fontweight="bold",
    pad=12
)

ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

# Grid and style
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
ax.legend(frameon=False, loc="lower right")

plt.tight_layout()
plt.savefig("figure2_modularity_early_late_dotplot.png", dpi=300, bbox_inches="tight")
plt.show()