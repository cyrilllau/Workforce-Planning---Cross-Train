from gurobipy import *
import numpy as np
import time

class CrossTrainingExtensive:
    def __init__(self, training_sections, B=0, alpha=4, w1=1, w2=0):
        # Input Data and Parameters
        self.training_sections = training_sections
        self.B = B
        self.alpha = alpha
        self.w1 = w1
        self.w2 = w2

        # Output of the Model
        self.solution = None
        self.updated_skill_matrices = {}
        self.training_plan = {}
        self.cross_training_used = 0

        # Internal variables
        self.model = None
        self.x_vars = []
        self.z_vars = []
        self.zp_vars = []
        self.ob1_expr = None
        self.ob2_expr = None
        self.ncrosstrain_used = None
        self.nworkers = []
        self.njobs = []
        self.sk_matrices = []

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

    def train(self):
        # Build the model
        start_time = time.time()
        mdl = Model('gurobi_model')
        mdl.setParam('OutputFlag', 0)
        self.model = mdl

        # Initialize lists
        Nsections = len(self.training_sections)
        self.x_vars = []
        self.z_vars = []
        self.zp_vars = []
        coeffs = []
        nsims = []
        self.njobs = []
        self.nworkers = []
        atts = []
        self.sk_matrices = []
        pref_matrices = []

        # Extract data for each section
        for s in range(Nsections):
            section = self.training_sections[s+1]
            nworker = section['Nworker']
            njob = section['Njob']
            nsim = section['Nsim']
            att = section['att']
            skmatrix = section['skmatrix']
            pref_matrix = section['pref_matrix']
            coeff = self.coeff_alpha(njob)

            coeffs.append(coeff)
            nsims.append(nsim)
            self.njobs.append(njob)
            self.nworkers.append(nworker)
            atts.append(att)
            self.sk_matrices.append(skmatrix)
            pref_matrices.append(pref_matrix)

            # Define variables
            x = mdl.addVars(nworker, njob * (nsim + 1), vtype=GRB.BINARY, name='x{}'.format(s+1))
            z = mdl.addVars(njob * nsim, vtype=GRB.CONTINUOUS, name='z{}'.format(s+1))
            zp = mdl.addVars(njob * nsim, vtype=GRB.CONTINUOUS, name='zp{}'.format(s+1))

            self.x_vars.append(x)
            self.z_vars.append(z)
            self.zp_vars.append(zp)

        # Constraints and Objective Function
        self.ob1_expr = LinExpr()
        self.ob2_expr = LinExpr()
        self.ncrosstrain_used = LinExpr()

        for k in range(Nsections):
            x = self.x_vars[k]
            z = self.z_vars[k]
            zp = self.zp_vars[k]
            coeff = coeffs[k]
            nsim = nsims[k]
            njob = self.njobs[k]
            nworker = self.nworkers[k]
            att = atts[k]
            skmatrix = self.sk_matrices[k]
            pref_matrix = pref_matrices[k]

            # Team leader and tag relief constraints
            for i in range(nworker):
                if njob > 2:
                    avg_x = quicksum(x[i, j] for j in range(njob - 2)) / (njob - 2)
                    mdl.addConstr(x[i, njob - 1] - avg_x <= 0)
                    mdl.addConstr(x[i, njob - 2] - avg_x <= 0)

            # Maintaining previous knowledge
            for i in range(nworker):
                for j in range(njob):
                    mdl.addConstr(x[i, j] >= skmatrix[i][j])

            # Assignment constraints
            for s_sim in range(1, nsim + 1):
                for i in range(nworker):
                    for j in range(njob):
                        mdl.addConstr(x[i, j + s_sim * njob] <= x[i, j])  # To be assigned, need to be trained

            # Attendance constraints
            for s_sim in range(1, nsim + 1):
                for i in range(nworker):
                    mdl.addConstr(quicksum(x[i, j + s_sim * njob] for j in range(njob)) <= att[i][s_sim - 1])  # Need to show up

            # One-to-one assignment constraints
            for s_sim in range(1, nsim + 1):
                for j in range(njob):
                    mdl.addConstr(quicksum(x[i, j + s_sim * njob] for i in range(nworker)) <= 1)  # One-to-one assignment

            # Linearization for objective
            for s_sim in range(1, nsim + 1):
                for j in range(njob):
                    index_z = njob * (s_sim - 1) + j

                    total_x = quicksum(x[i, jj] for i in range(nworker) for jj in range(s_sim * njob, (s_sim + 1) * njob))
                    total_pref_x = quicksum(pref_matrix[i][j % njob] * x[i, jj] for i in range(nworker) for jj in range(s_sim * njob, (s_sim + 1) * njob))

                    mdl.addConstr(z[index_z] >= coeff[j] * (njob - total_x - j))
                    mdl.addConstr(zp[index_z] >= coeff[j] * (njob - total_pref_x - j))

            # Relaxed constraint component
            self.ncrosstrain_used += quicksum(x[i, j] - skmatrix[i][j] for i in range(nworker) for j in range(njob))

            # Objective function components
            self.ob1_expr += quicksum(z[j] for j in range(njob * nsim)) / nsim
            self.ob2_expr += quicksum(zp[j] for j in range(njob * nsim)) / nsim

        # Total relaxed constraint
        mdl.addConstr(self.ncrosstrain_used <= self.B)

        # Objective function
        obj_expr = self.w1 * self.ob1_expr + self.w2 * self.ob2_expr
        mdl.setObjective(obj_expr, GRB.MINIMIZE)

        # Optimize
        mdl.optimize()
        end_time = time.time()
        print('Time spent:', end_time - start_time)

        self.extract_solution()

    def extract_solution(self):
        mdl = self.model
        if mdl.Status == GRB.OPTIMAL:
            # Extract the updated skill matrices and training plan
            for k in range(len(self.x_vars)):
                x = self.x_vars[k]
                nworker = self.nworkers[k]
                njob = self.njobs[k]
                skmatrix = self.sk_matrices[k]
                x_sol = mdl.getAttr('X', x)
                updated_skmatrix = np.copy(skmatrix)
                training_assignments = []

                for i in range(nworker):
                    for j in range(njob):
                        x_ij = x_sol[i, j]
                        if x_ij > skmatrix[i][j] + 0.5:  # Allow for floating point tolerance
                            updated_skmatrix[i][j] = 1
                            training_assignments.append((i, j))
                            self.cross_training_used += 1

                self.updated_skill_matrices[k+1] = updated_skmatrix
                self.training_plan[k+1] = training_assignments

            # Output results
            print("=== Optimal Solution ===")
            print(f"Optimal value of the objective function: {mdl.objVal:.4f}")
            print("=== Training Assignments ===")
            for k in range(len(self.training_plan)):
                nworker = self.nworkers[k]
                njob = self.njobs[k]
                training_assignments = self.training_plan[k+1]
                print(f"Section {k+1}:")
                if training_assignments:
                    for (i, j) in training_assignments:
                        print(f"Worker {i+1} is trained for Job {j+1}")
                else:
                    print("No new training assignments.")
                print()
        else:
            print("No feasible solution found.")
