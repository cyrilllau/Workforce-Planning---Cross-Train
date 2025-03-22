# run_varying_weights.py

import os
import numpy as np
from data_loader import DataLoader
from cross_training_model_extensive import CrossTrainingExtensive
from model_evaluator import ModelEvaluator

def main():
    # Initialize DataLoader
    train_ratio = 0.8  # 80% for training, 20% for testing
    data_loader = DataLoader(train_ratio)
    data_loader.load_data(sections=7)

    # Get training and testing data
    training_sections = data_loader.training_sections
    testing_sections = data_loader.testing_sections

    # Varying weight combinations
    weight_combinations = [
        (1, 0),
        (0.9, 0.1),
        (0.5, 0.5)
    ]
    results_dir = 'results/VaryingWeights'
    os.makedirs(results_dir, exist_ok=True)

    for w1, w2 in weight_combinations:
        unoccupied_file = os.path.join(results_dir, f'unoccupied_jobs_w1_{w1}_w2_{w2}.npy')
        preferences_file = os.path.join(results_dir, f'missed_preferences_w1_{w1}_w2_{w2}.npy')

        if os.path.exists(unoccupied_file) and os.path.exists(preferences_file):
            print(f"Results for w1 = {w1}, w2 = {w2} already exist. Skipping computation.")
            continue

        print(f"\nRunning model for w1 = {w1}, w2 = {w2}")
        # Initialize and train the model with default B = 4, alpha = 1
        cross_training_model = CrossTrainingExtensive(training_sections, B=8, alpha=1, w1=w1, w2=w2)
        cross_training_model.train()

        # Evaluate the model
        evaluator = ModelEvaluator(
            testing_sections=testing_sections,
            updated_skill_matrices=cross_training_model.updated_skill_matrices,
            alpha=1,
            w1=w1,
            w2=w2
        )
        evaluator.evaluate()

        # Extract the unoccupied jobs and missed preferences after cross-training
        unoccupied_after = np.array([evaluator.unoccupied_jobs_after[s] for s in sorted(evaluator.unoccupied_jobs_after.keys())])
        missed_preferences_after = np.array([evaluator.missed_preferences_after[s] for s in sorted(evaluator.missed_preferences_after.keys())])

        # Save the unoccupied jobs and missed preferences data
        np.save(unoccupied_file, unoccupied_after)
        np.save(preferences_file, missed_preferences_after)
        print(f"Results for w1 = {w1}, w2 = {w2} saved.")

if __name__ == "__main__":
    main()
