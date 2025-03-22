# plot_varying_alpha.py

import os
import numpy as np
import matplotlib.pyplot as plt
import math

# Enable LaTeX for matplotlib plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def main():
    # Varying alpha values
    alpha_values = [1, 2]
    results_dir = 'results/VaryingAlpha'

    # Prepare data storage
    unoccupied_jobs_daily = {}  # Unoccupied jobs after cross-training for each alpha

    for alpha in alpha_values:
        result_file = os.path.join(results_dir, f'unoccupied_jobs_alpha_{alpha}.npy')

        if os.path.exists(result_file):
            print(f"Loading saved results for alpha = {alpha}")
            unoccupied_after = np.load(result_file)
            unoccupied_jobs_daily[alpha] = unoccupied_after
        else:
            print(f"Results for alpha = {alpha} not found. Please run 'run_varying_alpha.py' first.")
            continue

    # Define styles for differentiation
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    markers = ['o', 's', '^', 'x']  # Circle, square, triangle, cross
    hatches = ['////', '***', '----', '++++', 'xxxx']

    # Aggregate over specified number of days to smooth the curve
    aggregate_days = 5  # Aggregate by 5 days (weekly data)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Prepare handles and labels for the combined legend
    handles = []
    labels = []

    # First plot: Line plot
    for idx, alpha in enumerate(alpha_values):
        unoccupied_after = unoccupied_jobs_daily[alpha]

        # Aggregate data
        total_days = len(unoccupied_after)
        unoccupied_after_agg = [
            np.sum(unoccupied_after[i:i+aggregate_days]) for i in range(0, total_days, aggregate_days)
        ]

        # Calculate weekly unoccupancy ratio
        total_jobs_per_week = 78 * aggregate_days  # Adjust this if the total number of jobs per week is different
        unoccupancy_ratio = [value / total_jobs_per_week for value in unoccupied_after_agg]

        # Weeks for x-axis
        aggregated_weeks = [i + 1 for i in range(len(unoccupied_after_agg))]

        # Plot
        h_line, = ax1.plot(aggregated_weeks, unoccupancy_ratio,
                           linestyle=line_styles[idx % len(line_styles)],
                           marker=markers[idx % len(markers)], markersize=6,
                           color='black',
                           label=rf'$\alpha={alpha}$')

        handles.append(h_line)
        labels.append(rf'$\alpha={alpha}$')

    ax1.set_xlabel('Week', fontsize=14)
    ax1.set_ylabel('Weekly Unoccupancy Ratio', fontsize=14)
    ax1.set_title('Weekly Unoccupancy Ratio Over Time', fontsize=16)
    ax1.grid(True, linestyle=':', linewidth=0.5)

    # Second plot: Bar plot
    max_unoccupied = max(max(unoccupied_jobs_daily[alpha]) for alpha in alpha_values)
    bins = range(0, int(max_unoccupied) + 2)  # +2 to include the last bin

    x = np.arange(len(bins) - 1)  # x positions
    width = 0.8 / len(alpha_values)  # width of the bars

    for idx, alpha in enumerate(alpha_values):
        unoccupied_after = unoccupied_jobs_daily[alpha]

        # Compute histogram
        hist_after, _ = np.histogram(unoccupied_after, bins=bins)

        # Convert to relative frequency
        total_days = len(unoccupied_after)
        rel_freq_after = hist_after / total_days

        # Adjust x positions
        x_positions = x - 0.4 + idx * width

        # Plot bars with different hatches
        b = ax2.bar(x_positions, rel_freq_after, width=width,
                    hatch=hatches[idx % len(hatches)], edgecolor='black', fill=False,
                    label=rf'$\alpha={alpha}$')

        handles.append(b)
        labels.append(rf'$\alpha={alpha}$')

    ax2.set_xlabel('Number of Unoccupied Jobs', fontsize=14)
    ax2.set_ylabel('Relative Frequency', fontsize=14)
    ax2.set_title('Relative Frequency of Unoccupied Jobs', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bins[:-1])
    ax2.grid(True, linestyle=':', linewidth=0.5)

    # Remove duplicate labels
    from collections import OrderedDict
    handles_labels = list(zip(handles, labels))
    handles_labels = list(OrderedDict.fromkeys(handles_labels))
    handles, labels = zip(*handles_labels)

    # Create a single, larger legend
    fig.legend(handles, labels, loc='upper center', fontsize=14, ncol=len(alpha_values))

    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Save and show the figure
    output_dir = 'figure/VaryingAlpha'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'combined_plot_varying_alpha.png'), dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
