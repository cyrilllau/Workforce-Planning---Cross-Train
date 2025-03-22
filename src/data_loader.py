import pandas as pd

class DataLoader:
    def __init__(self, train_ratio):
        # Dictionaries to store data for training and testing sections
        self.training_sections = {}
        self.testing_sections = {}
        self.train_ratio = train_ratio

    def read_data(self, filename):
        # Read data from different sheets in the provided Excel file
        att = pd.read_excel(filename, sheet_name='att_initial', header=None, engine='openpyxl').to_numpy()[:, 1:]
        total_periods = att.shape[1]
        nsim_train = int(total_periods * self.train_ratio)
        nsim_test = total_periods - nsim_train

        # Split attendance data into training and testing
        att_train = att[:, :nsim_train]
        att_test = att[:, nsim_train:]

        skmatrix = pd.read_excel(filename, sheet_name='vers', engine='openpyxl').to_numpy()[:, 1:]
        nworker, njob = skmatrix.shape

        pref_matrix = pd.read_excel(filename, sheet_name='prior', engine='openpyxl').to_numpy()[:, 1:]

        return att_train, nsim_train, att_test, nsim_test, skmatrix, nworker, njob, pref_matrix

    def load_data(self, sections=7):
        # Read data for sections 1 to the specified number (default is 7)
        for k in range(1, sections + 1):
            filename = f'data/sec{k}.xlsx'
            att_train, nsim_train, att_test, nsim_test, skmatrix, nworker, njob, pref_matrix = self.read_data(filename)
            # Store training data
            self.training_sections[k] = {
                'Nworker': nworker,
                'Njob': njob,
                'Nsim': nsim_train,
                'att': att_train,
                'skmatrix': skmatrix,
                'pref_matrix': pref_matrix
            }
            # Store testing data
            self.testing_sections[k] = {
                'Nworker': nworker,
                'Njob': njob,
                'Nsim': nsim_test,
                'att': att_test,
                'skmatrix': skmatrix,  # Initial skill matrix
                'pref_matrix': pref_matrix
            }
