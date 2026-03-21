
import os
import sys
import json
import numpy as np

# Attempt to import plotting libraries, provide helpful error if they are not installed.
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Error: Matplotlib or Seaborn is not installed.")
    print("Please install them using: pip install matplotlib seaborn")
    exit(1)

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
MATRICES_PATH = os.path.join(PROJECT_ROOT, 'prototypes', 'rich_similarity_matrices.json')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'plots', 'similarity_heatmaps')
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def create_rich_heatmap(matrix, city_name, output_path):
    """Generates and saves a heatmap for a given rich similarity matrix."""
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.1)
    
    ax = sns.heatmap(
        matrix,
        xticklabels=MONTH_LABELS,
        yticklabels=MONTH_LABELS,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=.5
    )
    
    ax.set_title(f'Rich Monthly Similarity Matrix for {city_name}', fontsize=18, pad=20)
    ax.set_xlabel('Late Period (1995-2024)', fontsize=14, labelpad=15)
    ax.set_ylabel('Early Period (1961-1990)', fontsize=14, labelpad=15)
    
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    print(f"  - Saved heatmap to {output_path}")

if __name__ == '__main__':
    # --- 1. Load the rich similarity matrices ---
    try:
        with open(MATRICES_PATH, 'r') as f:
            matrices = json.load(f)
    except FileNotFoundError:
        print(f"Error: Matrices file not found at {MATRICES_PATH}")
        print("Please run 'rich_prototype_analysis.py' first.")
        exit(1)

    # --- 2. Create output directory ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # --- 3. Generate a plot for each city ---
    for city, matrix_list in matrices.items():
        if not matrix_list:
            print(f"Skipping {city} due to empty matrix.")
            continue
        print(f"Processing city: {city}...")
        matrix = np.array(matrix_list)
        output_file_path = os.path.join(OUTPUT_DIR, f'rich_{city}_heatmap.png')
        create_rich_heatmap(matrix, city, output_file_path)

    print("\nAll rich heatmap visualizations have been generated.")
