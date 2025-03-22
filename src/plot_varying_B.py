# plot_varying_B.py

import os
import numpy as np
import matplotlib.pyplot as plt
import math

# Enable LaTeX for matplotlib plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def main():
    # Varying B values
    B_values = [0, 4, 8]
    results_dir = 'results/VaryingB'

    # Prepare data storage
    unoccupied_jobs_daily = {}  # Unoccupied jobs after cross-training for each B

    for B in B_values:
        result_file = os.path.join(results_dir, f'unoccupied_jobs_B_{B}.npy')

        if os.path.exists(result_file):
            print(f"Loading saved results for B = {B}")
            unoccupied_after = np.load(result_file)

            # Exclude the last 10 days
            unoccupied_after = unoccupied_after[:-14]

            unoccupied_jobs_daily[B] = unoccupied_after
        else:
            print(f"Results for B = {B} not found. Please run 'run_varying_B.py' first.")
            continue

    # Now plot the data
    # Plot unoccupied jobs over time for different B values in a single figure
    # Use different line styles and markers to differentiate between B values

    line_styles = ['-', '--', ':']
    markers = ['o', '^', 's']  # Circle, triangle, cross

    # Aggregate over specified number of days to smooth the curve
    aggregate_days = 5  # Aggregate by 5 days (weekly data)
    total_days = len(next(iter(unoccupied_jobs_daily.values())))
    num_intervals = math.ceil(total_days / aggregate_days)
    aggregated_weeks = [i + 1 for i in range(num_intervals)]  # Week numbers starting from 1

    plt.figure(figsize=(12, 6))

    for idx, B in enumerate(B_values):
        unoccupied_after = unoccupied_jobs_daily[B]

        # Aggregate data
        unoccupied_after_agg = [
            np.sum(unoccupied_after[i:i+aggregate_days]) for i in range(0, total_days, aggregate_days)
        ]

        # Calculate weekly unoccupancy ratio
        total_jobs_per_day = 78  # Total jobs per day
        total_jobs_per_period = total_jobs_per_day * aggregate_days  # Total jobs in the aggregation period
        unoccupancy_ratio = [value / total_jobs_per_period for value in unoccupied_after_agg]

        # Plot after cross-training
        plt.plot(aggregated_weeks, unoccupancy_ratio, linestyle=line_styles[idx % len(line_styles)],
                 marker=markers[idx % len(markers)], markersize=8, linewidth = 2.5 ,label=rf'$B={B}$')



    plt.xlabel('Week', fontsize=16)
    plt.ylabel('Weekly Unoccupancy Ratio', fontsize=16)
    #plt.title('Weekly Unoccupancy Ratio Over Time for Different $B$ Values', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('figure/VaryingB/weekly_unoccupancy_ratio_over_time.png', dpi=300)
    plt.show()

    # Bar plot for relative frequency
    hatches = ['////', '***', '----']
    plt.figure(figsize=(12, 6))

    # Combine all unoccupied_after arrays to find global max for bins
    all_unoccupied = np.concatenate([unoccupied_jobs_daily[B] for B in B_values])
    max_unoccupied = int(np.max(all_unoccupied))
    min_unoccupied = int(np.min(all_unoccupied))
    bins = range(min_unoccupied-2, max_unoccupied + 4)  # +2 to include the last bin

    x = np.arange(len(bins) - 1)  # x positions
    width = 0.8 / len(B_values)  # width of the bars

    for idx, B in enumerate(B_values):
        unoccupied_after = unoccupied_jobs_daily[B]

        # Compute histogram
        hist_after, _ = np.histogram(unoccupied_after, bins=bins)

        # Convert to relative frequency
        total_days = len(unoccupied_after)
        rel_freq_after = hist_after / total_days

        # Adjust x positions
        x_positions = x - 0.4 + idx * width

        # Plot bars with different hatches
        plt.bar(x_positions, rel_freq_after, width=width, hatch=hatches[idx % len(hatches)],
                edgecolor='black', fill=False, label=rf'$B={B}$')

    plt.xlabel('Number of Unoccupied Jobs', fontsize=16)
    plt.ylabel('Relative Frequency', fontsize=16)
    #plt.title('Relative Frequency of Unoccupied Jobs for Different $B$ Values', fontsize=16)
    plt.xticks(x, bins[:-1])
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('figure/VaryingB/relative_frequency_unoccupied_jobs.png', dpi=300)
    plt.show()

    # Line plot for unoccupied job distribution
    plt.figure(figsize=(12, 6))

    for idx, B in enumerate(B_values):
        unoccupied_after = unoccupied_jobs_daily[B]

        # Compute histogram
        hist_after, bin_edges = np.histogram(unoccupied_after, bins=bins)

        # Convert to relative frequency
        total_days = len(unoccupied_after)
        rel_freq_after = hist_after / total_days

        # Get the midpoints of bins for plotting
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot the distribution as a curve with markers
        plt.plot(bin_mids, rel_freq_after, linestyle=line_styles[idx % len(line_styles)],
                 marker=markers[idx % len(markers)], markersize=9, linewidth = 2.5 , label=rf'$B={B}$')

    plt.xlabel('Number of Unoccupied Jobs', fontsize=16)
    plt.ylabel('Relative Frequency', fontsize=16)
    #plt.title('Distribution of Unoccupied Jobs for Different $B$ Values', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('figure/VaryingB/unoccupied_jobs_distribution_curve.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
