# 单纯形法实现代码 
import numpy as np
from .models import LPProblemInput, LPSolution

# constants that represent the status of the solution
STATUS_OPTIMAL = "optimal"
STATUS_UNBOUNDED = "unbounded"
STATUS_INFEASIBLE = "infeasible" 
STATUS_NOT_IMPLEMENTED = "not_implemented"
STATUS_ERROR = "error"

# Tableau structure with Big M:
# Row 0: M-coefficients of reduced costs (d_j)
# Row 1: Regular-coefficients of reduced costs (c_j - z_j)
# Row 2 to num_constraints+1: Constraint rows
# Last column: RHS / Solution values
# Columns: Original vars, Slack vars, Surplus vars, Artificial vars, RHS

M_ROW = 0       # Index for M-coefficients in the objective function rows of the tableau
C_ROW = 1       # Index for regular coefficients
CONSTRAINTS_START_ROW = 2 # Constraint rows start after M-row and C-row

# Define a sufficiently large number for M, or handle symbolically if possible.
# For simplicity in direct numerical computation, using a large number.
# This can lead to precision issues. A symbolic approach is more robust.
BIG_M_VALUE = 1e7 # Placeholder, true Big M is symbolic

def solve_lp_problem(problem: LPProblemInput) -> LPSolution:
    """
    使用单纯形法求解线性规划问题。

    参数:
        problem (LPProblemInput): 包含线性规划问题定义的输入对象。

    返回:
        LPSolution: 包含求解结果的输出对象。
    """
    # 1. 参数校验
    if not problem.objective_coefficients or not problem.constraint_matrix or \
       len(problem.constraint_matrix) != len(problem.constraint_rhs) or \
       len(problem.constraint_matrix) != len(problem.constraint_senses):
        return LPSolution(status=STATUS_ERROR, message="input data format error or incomplete.")

    num_orig_vars = len(problem.objective_coefficients)
    num_constraints = len(problem.constraint_rhs)

    if any(len(row) != num_orig_vars for row in problem.constraint_matrix):
        return LPSolution(status=STATUS_ERROR, message="constraint matrix dimension does not match number of variables.")

    # --- Variable Categorization --- 
    # original_vars: 0 to num_orig_vars - 1
    # slack_vars: num_orig_vars to num_orig_vars + num_slack - 1
    # surplus_vars: num_orig_vars + num_slack to num_orig_vars + num_slack + num_surplus - 1
    # artificial_vars: num_orig_vars + num_slack + num_surplus to total_vars - 1

    senses = list(problem.constraint_senses)
    rhs = np.array(problem.constraint_rhs, dtype=float)
    A = np.array(problem.constraint_matrix, dtype=float)

    # Ensure RHS is non-negative
    for i in range(num_constraints):
        if rhs[i] < 0:
            rhs[i] *= -1
            A[i, :] *= -1
            if senses[i] == "<=": senses[i] = ">="
            elif senses[i] == ">=": senses[i] = "<="
            # '==' sense remains unchanged with row negation

    num_slack = sum(1 for s in senses if s == "<=")
    num_surplus = sum(1 for s in senses if s == ">=")
    num_artificial = sum(1 for s in senses if s == ">=" or s == "==")

    total_tableau_vars = num_orig_vars + num_slack + num_surplus + num_artificial
    
    # --- Tableau Initialization ---
    # Rows: M_coeff, C_coeff, constraint_rows
    # Cols: orig_vars, slack_vars, surplus_vars, artificial_vars, RHS
    tableau = np.zeros((CONSTRAINTS_START_ROW + num_constraints, total_tableau_vars + 1))
    
    # Store indices of different variable types
    slack_var_indices = []
    surplus_var_indices = []
    artificial_var_indices = []
    basis_vars_indices = [-1] * num_constraints # Stores the column index of the var in basis for each constraint row

    # Populate objective coefficients (for minimization problem)
    obj_coeffs_orig = np.array(problem.objective_coefficients, dtype=float)
    is_maximize = problem.objective_sense.lower() == "maximize"
    if is_maximize:
        obj_coeffs_orig *= -1 # Convert max to min

    # Objective function row (C_ROW) for original variables
    tableau[C_ROW, :num_orig_vars] = obj_coeffs_orig
    # M_ROW for original variables is 0, slack/surplus also 0

    # Populate constraints and identify initial basis variables
    current_slack_idx = num_orig_vars
    current_surplus_idx = num_orig_vars + num_slack
    current_artificial_idx = num_orig_vars + num_slack + num_surplus

    for i in range(num_constraints):
        tableau[CONSTRAINTS_START_ROW + i, :num_orig_vars] = A[i, :]
        tableau[CONSTRAINTS_START_ROW + i, total_tableau_vars] = rhs[i]

        if senses[i] == "<=":
            tableau[CONSTRAINTS_START_ROW + i, current_slack_idx] = 1
            slack_var_indices.append(current_slack_idx)
            basis_vars_indices[i] = current_slack_idx
            current_slack_idx += 1
        elif senses[i] == ">=":
            tableau[CONSTRAINTS_START_ROW + i, current_surplus_idx] = -1 # Surplus variable
            surplus_var_indices.append(current_surplus_idx)
            current_surplus_idx += 1
            
            tableau[CONSTRAINTS_START_ROW + i, current_artificial_idx] = 1 # Artificial variable
            artificial_var_indices.append(current_artificial_idx)
            basis_vars_indices[i] = current_artificial_idx
            tableau[M_ROW, current_artificial_idx] = 1 # M coefficient for artificial var in objective (min M*a_i)
            current_artificial_idx += 1
        elif senses[i] == "==":
            tableau[CONSTRAINTS_START_ROW + i, current_artificial_idx] = 1 # Artificial variable
            artificial_var_indices.append(current_artificial_idx)
            basis_vars_indices[i] = current_artificial_idx
            tableau[M_ROW, current_artificial_idx] = 1 # M coefficient for artificial var in objective
            current_artificial_idx += 1

    # Adjust initial objective rows (M_ROW, C_ROW) due to artificial variables in basis
    for i in range(num_constraints):
        basis_var_col = basis_vars_indices[i]
        if basis_var_col in artificial_var_indices: # If an artificial variable is in basis
            # We need to make its M-coefficient in M_ROW zero.
            # New M_ROW = Old M_ROW - 1 * (constraint_row_i containing that artificial var)
            # Since M_ROW only has 1s for artificial vars, and 0 for the one in basis
            # this effectively means: M_ROW = M_ROW - constraint_row_i
            # But we only do this for the objective function coefficients, not for RHS. 
            # And M_ROW already has 1 for this current_artificial_idx. So we subtract the constraint row from M_ROW.
            # More accurately: obj_M_coeffs = obj_M_coeffs - constraint_row
            # obj_C_coeffs = obj_C_coeffs - 0 * constraint_row (no change to C_ROW from M perspective)

            # For Big-M, initial tableau M_ROW for basis artificial variable is 1.
            # To make it 0: M_ROW = M_ROW - constraint_row_i
            # This also affects C_ROW: C_ROW = C_ROW - (value of c_j for that artificial variable, which is 0) * constraint_row
            # The value in tableau[M_ROW, basis_var_col] for an artificial var is 1 (cost of M)
            # We need to make these 0 for basic variables.
            # tableau[M_ROW,:] -= tableau[CONSTRAINTS_START_ROW + i, :] #This is the operation.
            # Correcting M_ROW and C_ROW based on artificial variables in basis
            # For each artificial variable a_k in the basis (coefficient of M is 1 in objective):
            # New M_obj_row = Old M_obj_row - 1 * (constraint_row_k)
            # New C_obj_row = Old C_obj_row - 0 * (constraint_row_k) (since c_k=0 for artificial vars)
            
            # The M_ROW has 1s at artificial variable columns. We need to make the M_ROW value 0 
            # for the artificial variables that are IN THE BASIS.
            # This is done by: New M_ROW = M_ROW - (Constraint Row of that basic artificial variable)
            # And: New C_ROW = C_ROW - (Original C_j of that basic artificial variable) * (Constraint Row)
            # Since C_j for artificial variables is 0, C_ROW is mainly affected by original variables that might be in basis. 
            # Here, initial basis is slack or artificial. C_j for slack is 0. C_j for artificial is 0. So C_ROW only modified by initial obj_coeffs. 

            # If an artificial variable a_k is basic in row r:
            # tableau[M_ROW, :] = tableau[M_ROW, :] - tableau[CONSTRAINTS_START_ROW + r, :]
            # This makes tableau[M_ROW, a_k_column] = 0, and updates other M-coeffs and M-obj value.
            # tableau[C_ROW, :] = tableau[C_ROW, :] - 0 * tableau[CONSTRAINTS_START_ROW + r, :] (no change here for C_ROW)

            # For each row i that has an artificial variable as basic:
            pivot_row_for_obj_calc = tableau[CONSTRAINTS_START_ROW + i, :]
            tableau[M_ROW, :] -= pivot_row_for_obj_calc # Subtract constraint row from M-coeff row
            # C_ROW is NOT directly modified by M-cost of artificial variables here, 
            # it was set by original problem's coefficients. If original variables were basic with non-zero costs,
            # C_ROW would need similar adjustment. But initial basis are slacks (cost 0) or artificials (cost 0 in C, cost M in M-obj).

    # Iteration Loop (Simplex Method)
    max_iterations = 2 * num_constraints + num_orig_vars # Heuristic for max iterations
    for iteration in range(max_iterations):
        # Determine entering variable
        # Priority to M-coefficients, then C-coefficients
        # Most negative d_j (M_ROW), if ties or all d_j >=0, then most negative c_j (C_ROW)
        
        min_m_coeff = 0
        min_c_coeff = 0
        pivot_col = -1

        # Check M_ROW first (coefficients of M in reduced costs)
        for j in range(total_tableau_vars):
            if tableau[M_ROW, j] < min_m_coeff - 1e-9:
                min_m_coeff = tableau[M_ROW, j]
                pivot_col = j
        
        if pivot_col == -1: # All M-coefficients are >= 0
            # Check C_ROW (regular reduced costs)
            for j in range(total_tableau_vars):
                if tableau[C_ROW, j] < min_c_coeff - 1e-9:
                    min_c_coeff = tableau[C_ROW, j]
                    pivot_col = j

        if pivot_col == -1: # All M-coeffs >= 0 and all C-coeffs >= 0 --> Optimal (or infeasible)
            # Check for infeasibility: if M-part of objective value is non-zero
            # Objective value = - (tableau[C_ROW, total_tableau_vars] + BIG_M_VALUE * tableau[M_ROW, total_tableau_vars])
            # If tableau[M_ROW, total_tableau_vars] is significantly negative (meaning M term is positive in obj func)
            if tableau[M_ROW, total_tableau_vars] < -1e-9: # M-part of objective is positive
                return LPSolution(status=STATUS_INFEASIBLE, message="original problem has no feasible solution (artificial variables > 0).")
            
            # Optimal solution found
            solution_vars = np.zeros(num_orig_vars)
            # Reconstruct solution for original variables
            for r in range(num_constraints):
                basis_var = basis_vars_indices[r]
                if basis_var < num_orig_vars: # If an original variable is in basis
                    solution_vars[basis_var] = tableau[CONSTRAINTS_START_ROW + r, total_tableau_vars]
            
            obj_val = -tableau[C_ROW, total_tableau_vars] # M-part should be (close to) zero here
            if is_maximize:
                obj_val *= -1

            # Check for infinite solutions (non-basic variable with zero reduced cost in both M and C rows)
            has_infinite_solutions = False
            for j in range(total_tableau_vars):
                is_basic = False
                for basis_idx in basis_vars_indices:
                    if j == basis_idx:
                        is_basic = True
                        break
                if not is_basic and np.isclose(tableau[M_ROW, j], 0) and np.isclose(tableau[C_ROW, j], 0):
                    has_infinite_solutions = True
                    break
            
            status_msg = STATUS_OPTIMAL
            if has_infinite_solutions:
                status_msg += " (multiple optimal solutions)"
            else:
                status_msg += " (only one optimal solution)"

            return LPSolution(status=status_msg, solution_variables=list(solution_vars), objective_value=obj_val)

        # Determine leaving variable (pivot row) using minimum ratio test
        min_ratio = float('inf')
        pivot_row = -1
        for i in range(num_constraints):
            # tableau_row_idx = CONSTRAINTS_START_ROW + i
            if tableau[CONSTRAINTS_START_ROW + i, pivot_col] > 1e-9: # Denominator must be positive
                ratio = tableau[CONSTRAINTS_START_ROW + i, total_tableau_vars] / tableau[CONSTRAINTS_START_ROW + i, pivot_col]
                if ratio < min_ratio - 1e-9:
                    min_ratio = ratio
                    pivot_row = CONSTRAINTS_START_ROW + i # Actual row index in tableau
                # TODO: Bland's rule for tie-breaking if ratio is same

        if pivot_row == -1:
            return LPSolution(status=STATUS_UNBOUNDED, message="problem is unbounded.") # Should check M-coeff of unbounded direction too

        # Perform pivot operation
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element # Normalize pivot row

        for i in range(CONSTRAINTS_START_ROW + num_constraints):
            if i != pivot_row:
                factor = tableau[i, pivot_col]
                tableau[i, :] -= factor * tableau[pivot_row, :]
        
        # Update objective rows (M_ROW and C_ROW)
        factor_m = tableau[M_ROW, pivot_col]
        tableau[M_ROW, :] -= factor_m * tableau[pivot_row, :]
        factor_c = tableau[C_ROW, pivot_col]
        tableau[C_ROW, :] -= factor_c * tableau[pivot_row, :]
        
        # Update basis variable for the pivot_row
        basis_vars_indices[pivot_row - CONSTRAINTS_START_ROW] = pivot_col

    return LPSolution(status=STATUS_ERROR, message=f"reached max iterations ({max_iterations}) without convergence.")


# Test cases need to be adapted for Big M or Two-Phase
if __name__ == "__main__":
    # Example that would require Big M ( >= constraint)
    # Max Z = 3x1 + 5x2
    # s.t. x1 <= 4
    #      2x2 <= 12
    #      3x1 + 2x2 >= 18  --> 3x1 + 2x2 - s3 + a1 = 18
    # x1,x2,s3,a1 >=0
    # Min -Z = -3x1 -5x2 + M a1

    print("Testing Big M Simplex (Conceptual - needs robust symbolic M or careful numeric M)")

    # Test Case 1: Requires artificial variable
    # Min Z = 2x1 + 3x2
    # s.t.  0.5x1 + 0.25x2 <= 4
    #       x1 + 3x2 >= 20      --> x1 + 3x2 - e2 + a2 = 20
    #       x1 + x2 == 10       --> x1 + x2 + a3 = 10
    # Min Z = 2x1 + 3x2 + M a2 + M a3
    problem1 = LPProblemInput(
        objective_coefficients=[2, 3],
        objective_sense="minimize",
        constraint_matrix=[
            [0.5, 0.25],
            [1, 3],
            [1, 1]
        ],
        constraint_senses=["<=", ">=", "=="],
        constraint_rhs=[4, 20, 10]
    )
    # Expected: x1=5, x2=5, Z=25
    solution1 = solve_lp_problem(problem1)
    print(f"Solution 1: Status={solution1.status}, ObjVal={solution1.objective_value}, Vars={solution1.solution_variables}, Msg={solution1.message}")

    # Test Case 2: Infeasible problem
    # Max Z = x1 + x2
    # s.t. x1 <= 1
    #      x2 <= 1
    #      x1 + x2 >= 3  --> x1 + x2 -e3 + a3 = 3
    # Min -Z = -x1 -x2 + M a3
    problem2 = LPProblemInput(
        objective_coefficients=[1, 1],
        objective_sense="maximize",
        constraint_matrix=[
            [1, 0],
            [0, 1],
            [1, 1]
        ],
        constraint_senses=["<=", "<=", ">="],
        constraint_rhs=[1, 1, 3]
    )
    solution2 = solve_lp_problem(problem2)
    print(f"Solution 2: Status={solution2.status}, ObjVal={solution2.objective_value}, Vars={solution2.solution_variables}, Msg={solution2.message}")

    # Test Case from previous set (should still work if only <=)
    problem3 = LPProblemInput(
                objective_coefficients=[3, 2],
                objective_sense="maximize",
                constraint_matrix=[
                    [1, 1],
                    [1, -1]
                ],
                constraint_senses=["<=", "<="],
                constraint_rhs=[4, 2]
            )
    solution3 = solve_lp_problem(problem3)
    print(f"Solution 3 (all <=): Status={solution3.status}, ObjVal={solution3.objective_value}, Vars={solution3.solution_variables}, Msg={solution3.message}")

    # Test unbounded with Big M (if possible)
    # Max Z = 2x1 + x2
    # s.t. x1 - x2 >= 10 --> x1 - x2 - e1 + a1 = 10
    #      x1 >= 5 --> x1 - e2 + a2 = 5
    # Min -Z = -2x1 -x2 + M a1 + M a2
    # This might become unbounded AFTER artificial variables are driven out.
    problem4 = LPProblemInput(
        objective_coefficients=[2,1],
        objective_sense="maximize",
        constraint_matrix=[
            [1,-1],
            [1,0]
        ],
        constraint_senses=[">=", ">="], # Will introduce artificial vars
        constraint_rhs=[1,1] # Small RHS to make it feasible quickly, then test unbounded for original vars
    )
    # If x1-x2 >= 1 and x1 >=1. If x1=1, -x2 >=0 => x2<=0. P = 2+x2. x2 can be -inf, so P is unbounded. (Incorrect reasoning, P can decrease)
    # If x1 fixed, say x1=10. Then 10-x2 >=1 => x2 <= 9. Max 20+x2, if x2=9, P=29. This setup seems bounded.
    # Let's use standard unbounded example: Max Z = 2x1 + x2 s.t. x1-x2<=10, 2x1<=40. Add a >= constraint x1 >= 1.
    problem4_unbounded_mod = LPProblemInput(
        objective_coefficients=[2, 1], 
        objective_sense="maximize",
        constraint_matrix=[
            [1, -1],
            [2, 0],
            [1, 0] 
        ],
        constraint_senses=["<=", "<=", ">="], # The >= constraint requires Big M
        constraint_rhs=[10, 40, 1]
    )
    solution4 = solve_lp_problem(problem4_unbounded_mod)
    print(f"Solution 4 (Unbounded with >=): Status={solution4.status}, ObjVal={solution4.objective_value}, Vars={solution4.solution_variables}, Msg={solution4.message}")

