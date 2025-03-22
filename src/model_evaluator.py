import numpy as np
from gurobipy import *

class ModelEvaluator:
    def __init__(self, testing_sections, updated_skill_matrices, alpha=4, w1=1, w2=0):
        self.testing_sections = testing_sections
        self.updated_skill_matrices = updated_skill_matrices
        self.alpha = alpha
        self.w1 = w1
        self.w2 = w2

        # For storing evaluation results
        # Keys: day index (s)
        self.unoccupied_jobs_before = {}  # Total unoccupied jobs per day before cross-training
        self.missed_preferences_before = {}  # Total missed preferences per day before cross-training
        self.unoccupied_jobs_after = {}  # Total unoccupied jobs per day after cross-training
        self.missed_preferences_after = {}  # Total missed preferences per day after cross-training

    def coeff_alpha(self, njob):
        coeff = np.zeros(njob)
        coeff[0] = 1
        for i in range(1, njob):
            v = np.repeat(i, njob)
            v1 = np.zeros(njob)
            v1[0] = v[0]
            for j in range(1, njob):
                v1[j] = v[j] - j
            coeff[i] = (i ** self.alpha) - np.sum(coeff[:njob] * v1[:njob])
        return coeff

    def evaluate(self):
        # Assume all sections have the same number of testing days
        nsim = next(iter(self.testing_sections.values()))['Nsim']

        for s in range(nsim):
            total_unoccupied_before = 0
            total_unoccupied_after = 0
            total_missed_pref_before = 0
            total_missed_pref_after = 0

            for k in sorted(self.testing_sections.keys()):
                section = self.testing_sections[k]
                nworker = section['Nworker']
                njob = section['Njob']
                att = section['att']
                pref_matrix = section['pref_matrix']
                skmatrix_initial = section['skmatrix']
                skmatrix_updated = self.updated_skill_matrices.get(k, skmatrix_initial)

                coeff = self.coeff_alpha(njob)

                att_scenario = att[:, s]

                # Before cross-training
                unoccupied_before, missed_pref_before = self.solve_daily_assignment(
                    nworker, njob, att_scenario, skmatrix_initial, pref_matrix, coeff)
                total_unoccupied_before += unoccupied_before
                total_missed_pref_before += missed_pref_before

                # After cross-training
                unoccupied_after, missed_pref_after = self.solve_daily_assignment(
                    nworker, njob, att_scenario, skmatrix_updated, pref_matrix, coeff)
                total_unoccupied_after += unoccupied_after
                total_missed_pref_after += missed_pref_after

            # Store totals for day s
            self.unoccupied_jobs_before[s] = total_unoccupied_before
            self.unoccupied_jobs_after[s] = total_unoccupied_after
            self.missed_preferences_before[s] = total_missed_pref_before
            self.missed_preferences_after[s] = total_missed_pref_after

    def solve_daily_assignment(self, nworker, njob, att_scenario, skmatrix, pref_matrix, coeff):
        mdl = Model('DailyAssignment')
        mdl.setParam('OutputFlag', 0)

        # Decision variables
        x = mdl.addVars(nworker, njob, vtype=GRB.BINARY, name='x')

        # Penalty variables y[n] for unassigned jobs
        y_vars = mdl.addVars(njob, vtype=GRB.CONTINUOUS, name='y')

        # Penalty variables z[q] for missed preferences
        z_vars = mdl.addVars(njob, vtype=GRB.CONTINUOUS, name='z')

        # Objective function coefficients
        c_nk = coeff  # c_nk = coeff[n]
        d_qk = coeff  # Assuming d_{q,k} is the same as c_{n,k}

        # Objective function
        mdl.setObjective(
            self.w1 * quicksum(c_nk[n] * y_vars[n] for n in range(njob)) +
            self.w2 * quicksum(d_qk[q] * z_vars[q] for q in range(njob)),
            GRB.MINIMIZE)

        # Constraints
        # Workers' attendance constraints
        for i in range(nworker):
            mdl.addConstr(quicksum(x[i, j] for j in range(njob)) <= att_scenario[i])

        # Skill constraints
        for i in range(nworker):
            for j in range(njob):
                mdl.addConstr(x[i, j] <= skmatrix[i][j])

        # Job assignment constraints
        for j in range(njob):
            mdl.addConstr(quicksum(x[i, j] for i in range(nworker)) <= 1)

        # y[n] constraints for unassigned jobs penalty
        total_assigned = quicksum(x[i, j] for i in range(nworker) for j in range(njob))
        for n in range(njob):
            mdl.addConstr(y_vars[n] >= njob - total_assigned - n)
            mdl.addConstr(y_vars[n] >= 0)

        # z[q] constraints for missed preferences penalty
        total_preferences = quicksum(pref_matrix[i][j] * x[i, j] for i in range(nworker) for j in range(njob))
        for q in range(njob):
            mdl.addConstr(z_vars[q] >= njob - total_preferences - q)
            mdl.addConstr(z_vars[q] >= 0)

        # Optimize the model
        mdl.optimize()

        if mdl.Status == GRB.OPTIMAL:
            # The number of unoccupied jobs is y_vars[0].X
            unoccupied_jobs = y_vars[0].X
            # The number of missed preferences is z_vars[0].X
            missed_preferences = z_vars[0].X
            return unoccupied_jobs, missed_preferences
        else:
            print("No solution found for this scenario.")
            # Return zero to ensure aggregation works correctly
            return 0, 0
