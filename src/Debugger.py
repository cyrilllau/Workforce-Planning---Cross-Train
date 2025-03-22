from data_loader import DataLoader
from cross_training_model_extensive import CrossTrainingExtensive
from model_evaluator import ModelEvaluator
from plotter import Plotter
import matplotlib.pyplot as plt
import numpy as np
import math  # Import math module for ceiling function

def main():
    # Initialize DataLoader
    train_ratio = 0.8  # 80% for training, 20% for testing
    data_loader = DataLoader(train_ratio)
    data_loader.load_data(sections=7)

    # Get training and testing data
    training_sections = data_loader.training_sections
    testing_sections = data_loader.testing_sections

    # Varying B values
    B_values = [0, 2, 4, 8]

    # Prepare data storage
    unoccupied_jobs_daily = {}  # unoccupied jobs after cross-training for each B

    # For storing the total unoccupied jobs per day for each B
    total_days = None  # We'll set this after first evaluation

    for B in B_values:
        print(f"\nTesting with B = {B}")
        # Initialize and train the model
        cross_training_model = CrossTrainingExtensive(training_sections, B=B, alpha=4, w1=1, w2=0)
        cross_training_model.train()

        # Evaluate the model
        evaluator = ModelEvaluator(
            testing_sections=testing_sections,
            updated_skill_matrices=cross_training_model.updated_skill_matrices,
            alpha=4,
            w1=1,
            w2=0
        )
        evaluator.evaluate()

        # Collect data
        unoccupied_after = []

        total_days = len(evaluator.unoccupied_jobs_after)
        for s in range(total_days):
            unoccupied_after.append(evaluator.unoccupied_jobs_after[s])

        unoccupied_jobs_daily[B] = unoccupied_after

    # Now plot the data
    # Plot unoccupied jobs over time for different B values in a single figure
    # Use different line styles to differentiate between B values

    line_styles = ['-', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    # Extend line_styles if more styles are needed

    # Aggregate over specified number of days to smooth the curve
    aggregate_days = 2
    num_intervals = math.ceil(total_days / aggregate_days)
    aggregated_days = [i * aggregate_days + 1 for i in range(num_intervals)]

    plt.figure(figsize=(12, 6))

    for idx, B in enumerate(B_values):
        unoccupied_after = unoccupied_jobs_daily[B]

        # Aggregate data
        unoccupied_after_agg = [
            np.mean(unoccupied_after[i:i+aggregate_days]) for i in range(0, len(unoccupied_after), aggregate_days)
        ]

        # Plot after cross-training
        plt.plot(aggregated_days, unoccupied_after_agg, linestyle=line_styles[idx % len(line_styles)], label=f'B={B}')

    plt.xlabel('Day')
    plt.ylabel('Daily Number of Unoccupied Jobs')
    plt.title('Unoccupied Jobs Over Time for Different B Values')
    plt.legend()
    plt.show()

    # Bar plot for relative frequency
    hatches = ['////', '||||', '----', '++++', 'xxxx', '****']
    plt.figure(figsize=(12, 6))

    max_unoccupied = max(max(unoccupied_jobs_daily[B]) for B in B_values)
    bins = range(0, int(max_unoccupied) + 2)  # +2 to include the last bin

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
        plt.bar(x_positions, rel_freq_after, width=width, hatch=hatches[idx % len(hatches)], edgecolor='black', fill=False, label=f'B={B}')

    plt.xlabel('Number of Unoccupied Jobs')
    plt.ylabel('Relative Frequency')
    plt.title('Relative Frequency of Unoccupied Jobs for Different B Values')
    plt.xticks(x, bins[:-1])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()




