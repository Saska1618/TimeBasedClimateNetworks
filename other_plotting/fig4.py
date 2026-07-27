import pandas as pd
import matplotlib.pyplot as plt

# Data
df = pd.DataFrame({
    "Location": [
        "Brașov", "Deva", "Pécs", "Košice", "Kecskemét",
        "Cluj-Napoca", "Győr", "Oradea", "Keszthely", "Gheorgheni"
    ],
    "Early_threshold": [0.971, 0.972, 0.965, 0.961, 0.971, 0.972, 0.957, 0.962, 0.960, 0.960],
    "Late_threshold":  [0.957, 0.961, 0.958, 0.957, 0.967, 0.969, 0.957, 0.966, 0.964, 0.970],
    "I_perc": [-0.01419, -0.01060, -0.00759, -0.00390, -0.00364,
               -0.00349,  0.00000,  0.00370,  0.00402,  0.01002]
})

# Sort by percolation index for better readability
df = df.sort_values("I_perc").reset_index(drop=True)

# Colors: negative vs positive
colors = ["firebrick" if v < 0 else "steelblue" for v in df["I_perc"]]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(df["Location"], df["I_perc"], color=colors)

# Zero line
ax.axhline(0, color="black", linewidth=1.0)

# Labels and title
ax.set_xlabel("Location")
ax.set_ylabel("Percolation index (I_perc)")
ax.set_title(
    "Percolation index of the time-based climate networks by location.",
    fontweight="bold",
    pad=12
)

# Style
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Rotate x labels
plt.xticks(rotation=45, ha="right")

# Value labels
for bar, val in zip(bars, df["I_perc"]):
    if val >= 0:
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.0005, f"{val:.5f}",
                ha="center", va="bottom", fontsize=9)
    else:
        ax.text(bar.get_x() + bar.get_width()/2, val - 0.0005, f"{val:.5f}",
                ha="center", va="top", fontsize=9)

plt.tight_layout()
plt.savefig("figure5_percolation_index_by_location.png", dpi=300, bbox_inches="tight")
plt.show()