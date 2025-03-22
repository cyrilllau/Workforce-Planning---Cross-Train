# run_varying_B.py

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

    # Varying B values
    B_values = [0, 4, 8]
    results_dir = 'results/VaryingB'
    os.makedirs(results_dir, exist_ok=True)

    for B in B_values:
        result_file = os.path.join(results_dir, f'unoccupied_jobs_B_{B}.npy')

        # if os.path.exists(result_file):
        #     print(f"Results for B = {B} already exist. Skipping computation.")
        #     continue

        print(f"\nRunning model for B = {B}")
        # Initialize and train the model
        cross_training_model = CrossTrainingExtensive(training_sections, B=B, alpha=1, w1=1, w2=0)
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

        # Extract the unoccupied jobs after cross-training
        unoccupied_after = np.array([evaluator.unoccupied_jobs_after[s] for s in sorted(evaluator.unoccupied_jobs_after.keys())])

        # Save the unoccupied jobs data
        np.save(result_file, unoccupied_after)
        print(f"Results for B = {B} saved.")

if __name__ == "__main__":
    main()
