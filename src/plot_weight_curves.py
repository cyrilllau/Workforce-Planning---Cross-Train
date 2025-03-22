# plot_varying_weights.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ------------------------------
# LaTeX Configuration for Matplotlib
# ------------------------------
# Enable LaTeX rendering for text in plots
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX for all text
    "font.family": "serif",  # Use serif fonts (e.g., Times New Roman)
    "font.size": 14,  # Set global font size
    "axes.labelsize": 14,  # Font size for axis labels
    "axes.titlesize": 16,  # Font size for titles
    "legend.fontsize": 12,  # Font size for legends
    "xtick.labelsize": 12,  # Font size for x-tick labels
    "ytick.labelsize": 12  # Font size for y-tick labels
})


def main():
    # Define the weight combinations for which to plot data
    weight_combinations = [
        (1, 0),
        (0.9, 0.1),
        (0.5, 0.5)
    ]
    results_dir = 'results/VaryingWeights'

    # Prepare data storage
    unoccupied_jobs_daily = {}  # Unoccupied jobs after cross-training for each weight combination
    missed_preferences_daily = {}  # Missed preferences after cross-training for each weight combination

    # Load data for each specified weight combination
    for w1, w2 in weight_combinations:
        unoccupied_file = os.path.join(results_dir, f'unoccupied_jobs_w1_{w1}_w2_{w2}.npy')
        preferences_file = os.path.join(results_dir, f'missed_preferences_w1_{w1}_w2_{w2}.npy')

        if os.path.exists(unoccupied_file) and os.path.exists(preferences_file):
            print(f"Loading saved results for w1 = {w1}, w2 = {w2}")
            try:
                unoccupied_after = np.load(unoccupied_file)
                unoccupied_after = unoccupied_after[:-14]
                missed_preferences_after = np.load(preferences_file)
                unoccupied_jobs_daily[(w1, w2)] = unoccupied_after
                missed_preferences_after = missed_preferences_after[:-14]
                missed_preferences_daily[(w1, w2)] = missed_preferences_after
            except Exception as e:
                print(f"Error loading files for w1 = {w1}, w2 = {w2}: {e}")
                continue
        else:
            print(f"Results for w1 = {w1}, w2 = {w2} not found. Please run 'run_varying_weights.py' first.")
            continue

    # Check if any data was loaded
    if not unoccupied_jobs_daily:
        print("No data loaded. Exiting script.")
        return



    # Define line styles for better distinguishability
    line_styles = ['-', '--', ':']  # Solid, dashed, dash-dot
    markers = ['o', '^', 's']  # Circle, triangle, cross

    # Aggregate over specified number of days to smooth the curve
    aggregate_days = 5  # Aggregate by 5 days (weekly data)

    # Determine the maximum length of data to handle varying lengths
    max_length = max(len(data) for data in unoccupied_jobs_daily.values())
    num_intervals = math.ceil(max_length / aggregate_days)
    aggregated_weeks = [i + 1 for i in range(num_intervals)]  # Week numbers starting from 1

    # Ensure the output directory exists
    output_dir = 'figure/VaryingWeights'
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------
    # Plot for Unoccupied Jobs Over Time (Overtime Curve with Markers)
    # ------------------------------
    plt.figure(figsize=(12, 6))
    for idx, (w1, w2) in enumerate(weight_combinations):
        if (w1, w2) not in unoccupied_jobs_daily:
            continue
        unoccupied_after = unoccupied_jobs_daily[(w1, w2)]

        # Aggregate data
        unoccupied_after_agg = [
            np.sum(unoccupied_after[i:i + aggregate_days]) for i in range(0, len(unoccupied_after), aggregate_days)
        ]

        # Calculate weekly unoccupancy ratio
        total_jobs_per_week = 78  # Adjust this if the total number of jobs per week is different
        unoccupancy_ratio = [value / total_jobs_per_week for value in unoccupied_after_agg]

        # Plot after cross-training with markers
        plt.plot(
            aggregated_weeks[:len(unoccupancy_ratio)],
            unoccupancy_ratio,
            linestyle=line_styles[idx % len(line_styles)],
            marker=markers[idx % len(markers)],  # Add a consistent marker
            markersize=8,  # Marker size for visibility
            linewidth=2.5,  # Line width for visibility
            #color='black',
            label=rf'$w_1={w1}, w_2={w2}$'
        )

    plt.xlabel('Week', fontsize=16)
    plt.ylabel('Weekly Unoccupancy Ratio', fontsize=16)
   # plt.title('Weekly Unoccupancy Ratio Over Time for Different Weight Combinations', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weekly_unoccupancy_ratio_over_time.png'), dpi=300)
    plt.show()

    # ------------------------------
    # Plot for Missed Preferences Over Time (Overtime Curve with Markers)
    # ------------------------------
    plt.figure(figsize=(12, 6))
    for idx, (w1, w2) in enumerate(weight_combinations):
        if (w1, w2) not in missed_preferences_daily:
            continue
        missed_preferences_after = missed_preferences_daily[(w1, w2)]

        # Aggregate data
        missed_prefs_agg = [
            np.sum(missed_preferences_after[i:i + aggregate_days]) for i in
            range(0, len(missed_preferences_after), aggregate_days)
        ]

        # Calculate weekly missed preferences ratio
        total_preferences_per_week = 78  # Assuming one preference per job per day
        missed_prefs_ratio = [value / total_preferences_per_week for value in missed_prefs_agg]

        # Plot after cross-training with markers
        plt.plot(
            aggregated_weeks[:len(missed_prefs_ratio)],
            missed_prefs_ratio,
            linestyle=line_styles[idx % len(line_styles)],
            marker=markers[idx % len(markers)],  # Add a consistent marker,
            markersize=8,  # Marker size for visibility
            linewidth=2.5,  # Line width for visibility
            #color='black',
            label=rf'$w_1={w1}, w_2={w2}$'
        )

    plt.xlabel('Week', fontsize=16)
    plt.ylabel('Weekly Missed Priority Assignment Ratio', fontsize=16)
    #plt.title('Weekly Missed Preferences Ratio Over Time for Different Weight Combinations', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weekly_missed_preferences_ratio_over_time.png'), dpi=300)
    plt.show()

    # ------------------------------
    # Distribution Curves for Unoccupied Jobs with Lines Only
    # ------------------------------
    plt.figure(figsize=(12, 6))
    for idx, (w1, w2) in enumerate(weight_combinations):
        if (w1, w2) not in unoccupied_jobs_daily:
            continue
        unoccupied_after = unoccupied_jobs_daily[(w1, w2)]

        # Define bins for histogram if not already defined
        max_unoccupied = int(np.max(unoccupied_after))
        min_unoccupied = int(np.min(unoccupied_after))
        bins_unoccupied = range(min_unoccupied-2, max_unoccupied + 4)  # +2 to include the last bin

        # Compute histogram
        hist_after, bin_edges = np.histogram(unoccupied_after, bins=bins_unoccupied)

        # Convert to relative frequency
        total_days = len(unoccupied_after)
        rel_freq_after = hist_after / total_days

        # Get the midpoints of bins for plotting
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot the distribution as a curve without markers
        plt.plot(
            bin_mids,
            rel_freq_after,
            linestyle=line_styles[idx % len(line_styles)],
            marker=markers[idx % len(markers)],  # Add a consistent marker,
            markersize=8,
            linewidth=2.5,  # Increased line width for better visibility
            #color='black',
            label=rf'$w_1={w1}, w_2={w2}$'
        )

    plt.xlabel('Number of Unoccupied Jobs', fontsize=16)
    plt.ylabel('Relative Frequency', fontsize=16)
    #plt.title('Distribution of Unoccupied Jobs for Different Weight Combinations', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'unoccupied_jobs_distribution_curve.png'), dpi=300)
    plt.show()

    # ------------------------------
    # Distribution Curves for Missed Preferences with Lines Only (Starting from 5)
    # ------------------------------
    plt.figure(figsize=(12, 6))
    for idx, (w1, w2) in enumerate(weight_combinations):
        if (w1, w2) not in missed_preferences_daily:
            continue
        missed_preferences_after = missed_preferences_daily[(w1, w2)]

        # Filter data to include only missed preferences >= 5
        missed_preferences_filtered = missed_preferences_after[missed_preferences_after >= 5]

        # Check if there is any data after filtering
        if len(missed_preferences_filtered) == 0:
            print(f"No missed preferences >= 4 for w1={w1}, w2={w2}. Skipping plot.")
            continue

        # Define bins for histogram starting from 5
        max_missed_prefs = int(np.max(missed_preferences_filtered))
        min_missed_prefs = int(np.min(missed_preferences_filtered))
        bins_prefs = range(min_missed_prefs - 2, max_missed_prefs + 4)  # Start from 5

        # Compute histogram
        hist_prefs, bin_edges_prefs = np.histogram(missed_preferences_filtered, bins=bins_prefs)

        # Convert to relative frequency
        total_days = len(missed_preferences_filtered)
        rel_freq_prefs = hist_prefs / total_days

        # Get the midpoints of bins for plotting
        bin_mids_prefs = (bin_edges_prefs[:-1] + bin_edges_prefs[1:]) / 2



        # Plot the distribution as a curve without markers
        plt.plot(
            bin_mids_prefs,
            rel_freq_prefs,
            linestyle=line_styles[idx % len(line_styles)],
            marker=markers[idx % len(markers)],  # Add a consistent marker,
            markersize=8,
        linewidth=2.5,  # Increased line width for better visibility
            #color='black',
            label=rf'$w_1={w1}, w_2={w2}$'
        )

    plt.xlabel('Number of Missed Priority Assignments', fontsize=16)
    plt.ylabel('Relative Frequency', fontsize=16)
    #plt.title('Distribution of Missed Preferences for Different Weight Combinations', fontsize=16)
    plt.legend(fontsize = 14)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missed_preferences_distribution_curve.png'), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
