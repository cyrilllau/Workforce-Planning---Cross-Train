# plot_pie_chart_varying_alpha.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch

def main():
    # Define the alpha values for which to plot pie charts
    alpha_values = [1, 4]
    results_dir = 'results/VaryingAlpha'

    # Initialize a dictionary to store unoccupied jobs data for each alpha
    unoccupied_jobs_daily = {}

    # Load data for each specified alpha
    for alpha in alpha_values:
        result_file = os.path.join(results_dir, f'unoccupied_jobs_alpha_{alpha}.npy')

        if os.path.exists(result_file):
            print(f"Loading saved results for alpha = {alpha}")
            try:
                unoccupied_after = np.load(result_file)
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
                continue

            # Exclude the last 14 days from the data
            if len(unoccupied_after) <= 14:
                print(f"Warning: Not enough data for alpha = {alpha} after excluding last 14 days.")
                continue
            unoccupied_after = unoccupied_after[:-14]

            unoccupied_jobs_daily[alpha] = unoccupied_after
        else:
            print(f"Results for alpha = {alpha} not found. Please run 'run_varying_alpha.py' first.")
            continue

    # Check if any data was loaded
    if not unoccupied_jobs_daily:
        print("No data loaded. Exiting script.")
        return

    # Define the categories as per user requirement
    categories = ['0-4', '5-8', '9+']

    # Set Seaborn style for better aesthetics
    sns.set(style='whitegrid')

    # Define the "Set2" color palette with at least 6 colors
    set2_palette_full = sns.color_palette("Set2", n_colors=8)  # Ensuring enough colors
    # Select first, sixth, and second colors
    custom_colors = [set2_palette_full[0], set2_palette_full[6], set2_palette_full[1]]  # Green, Yellow, Red

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Loop through each alpha value to create pie charts
    for idx, alpha in enumerate(alpha_values):
        if alpha not in unoccupied_jobs_daily:
            print(f"No data for alpha = {alpha}. Skipping pie chart.")
            continue

        unoccupied_after = unoccupied_jobs_daily[alpha]

        # Initialize counts for each category
        category_counts = [0, 0, 0]

        # Categorize the unoccupied jobs data
        for value in unoccupied_after:
            if 0 <= value <= 4:
                category_counts[0] += 1
            elif 5 <= value <= 8:
                category_counts[1] += 1
            elif value >= 9:
                category_counts[2] += 1

        # Create a DataFrame for plotting
        df = pd.DataFrame({
            'Category': categories,
            'Counts': category_counts
        })

        # Calculate percentages for each category
        total_days = len(unoccupied_after)
        df['Percentage'] = (df['Counts'] / total_days) * 100

        # Define explode parameter; no explosion to keep slices uniform
        explode = (0, 0, 0)

        # Create a pie chart in the subplot
        ax = axes[idx]
        wedges, texts, autotexts = ax.pie(
            df['Counts'],
            labels=None,                             # We will manually add labels
            colors=custom_colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else '',
            startangle=90,
            pctdistance=0.8,                        # Position percentage labels closer to the center
            labeldistance=1.05,                      # Position category labels closer to the pie chart
            explode=explode,
            textprops={
                'fontsize': 14,                      # Slightly increased font size
                'color': 'black',
                'fontweight': 'bold'                 # Make text bold
            },
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )

        # Manually add category labels to control their positions
        for i, (wedge, category) in enumerate(zip(wedges, categories)):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))
            horizontalalignment = 'left' if x > 0 else 'right'
            connectionstyle = f"angle,angleA=0,angleB={angle}"
            kw = dict(
                xycoords='data',
                textcoords='data',
                arrowprops=dict(arrowstyle="-", color='black', connectionstyle=connectionstyle),
                horizontalalignment=horizontalalignment,
                fontsize=14,                      # Slightly increased font size
                color='black',
                fontweight='bold'                 # Make text bold
            )
            # Adjust label positions by scaling x and y
            x_label = 1.05 * x
            y_label = 1.05 * y
            ax.annotate(category, xy=(x, y), xytext=(x_label, y_label), **kw)

        # Adjust percentage labels font size and properties
        for autotext in autotexts:
            autotext.set_fontsize(14)              # Slightly increased font size
            autotext.set_color('black')
            autotext.set_fontweight('bold')        # Make text bold

        # Draw a circle at the center to make it a donut chart
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax.add_artist(centre_circle)

        # Ensure the pie chart is drawn as a circle
        ax.axis('equal')

        # Add title to the pie chart with larger font
        ax.set_title(f'α = {alpha}', fontsize=18, weight='bold')

    # Create custom legend entries using the colors from the pie chart
    legend_elements = [
        Patch(facecolor=color, label=label)
        for color, label in zip(custom_colors, categories)
    ]
    # Place the legend outside the subplots
    fig.legend(handles=legend_elements, title="Categories", loc='upper center',
               fontsize=16, title_fontsize=16, ncol=3, frameon=False)

    # Adjust layout to accommodate legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)  # Adjusted top to make room for the smaller legend

    # Save and display the pie charts
    output_dir = 'figure/VaryingAlpha'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'pie_charts_alpha_1_and_4.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Pie charts saved to {output_path}")

    plt.show()

if __name__ == "__main__":
    main()
