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
    B_values = [0, 8]
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



    # Bar plot for relative frequency
    hatches = ['////', '***', '----']
    plt.figure(figsize=(12, 6))

    # Combine all unoccupied_after arrays to find global max for bins
    all_unoccupied = np.concatenate([unoccupied_jobs_daily[B] for B in B_values])
    max_unoccupied = int(np.max(all_unoccupied))
    min_unoccupied = int(np.min(all_unoccupied))
    bins = range(min_unoccupied-1, max_unoccupied +3)  # +2 to include the last bin

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
        if B == 0:

            plt.bar(x_positions, rel_freq_after, width=width,
                    #hatch=hatches[idx % len(hatches)],
                    edgecolor='black',
                    #fill=False,
                    label='Before Cross Training')

        else:
            plt.bar(x_positions, rel_freq_after, width=width,
                    #hatch=hatches[idx % len(hatches)],
                    edgecolor='black',
                    #fill=False,
                    label='After Cross Training')

    plt.xlabel('Number of Unoccupied Jobs', fontsize=14)
    plt.ylabel('Relative Frequency', fontsize=14)
    plt.xlim(1,13)
    #plt.title('Relative Frequency of Unoccupied Jobs for Different $B$ Values', fontsize=16)
    plt.xticks(x, bins[:-1])
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('figure/VaryingB/relative_frequency_unoccupied_jobs_before_after.png', dpi=300)
    plt.show()




    # Line plot for unoccupied job distribution
    line_styles = ['-', '-.']
    markers = ['s', '^']

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

        if B==0:

            plt.plot(bin_mids, rel_freq_after, linestyle=line_styles[idx % len(line_styles)],
                 marker=markers[idx % len(markers)], markersize=9,
                 linewidth = 2.5 , label='Before Cross Training')

        else:
            plt.plot(bin_mids, rel_freq_after, linestyle=line_styles[idx % len(line_styles)],
                     marker=markers[idx % len(markers)], markersize=9,
                     linewidth=2.5, label='After Cross Training')

    plt.xlabel('Number of Unoccupied Jobs', fontsize=16)
    plt.ylabel('Relative Frequency', fontsize=16)
    #plt.title('Distribution of Unoccupied Jobs for Different $B$ Values', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('figure/VaryingB/unoccupied_jobs_distribution_curve_before_after.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
