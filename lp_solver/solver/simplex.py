# Two-phase simplex method implementation.
import numpy as np

from .models import LPProblemInput, LPSolution
from .simplex_core import (
    TOL,
    CONSTRAINTS_START_ROW,
    COST_ROW,
    build_standard_form,
    run_simplex,
    extract_solution,
)

# constants that represent the status of the solution
STATUS_OPTIMAL = "optimal"
STATUS_UNBOUNDED = "unbounded"
STATUS_INFEASIBLE = "infeasible"
STATUS_NOT_IMPLEMENTED = "not_implemented"
STATUS_ERROR = "error"


def _validate(problem: LPProblemInput):
    """Return an error message string if invalid, else None."""
    if not isinstance(problem.objective_sense, str):
        return f"objective_sense must be a string, got: {type(problem.objective_sense).__name__}"
    sense = problem.objective_sense.strip().lower()
    if sense not in ("maximize", "minimize"):
        return f"objective_sense must be 'maximize' or 'minimize', got: {problem.objective_sense!r}"

    if not problem.objective_coefficients:
        return "problem has no variables (objective_coefficients is empty)"
    num_orig_vars = len(problem.objective_coefficients)
    if not problem.constraint_matrix:
        return "problem has no constraints (constraint_matrix is empty)"
    if len(problem.constraint_matrix) != len(problem.constraint_rhs) or \
       len(problem.constraint_matrix) != len(problem.constraint_senses):
        return "constraint_matrix, constraint_rhs, and constraint_senses must have equal length"

    try:
        obj = np.array(problem.objective_coefficients, dtype=float)
    except (ValueError, TypeError):
        return "objective_coefficients must be numeric"
    if not np.all(np.isfinite(obj)):
        return "objective_coefficients must be finite (no NaN or Inf)"

    for row in problem.constraint_matrix:
        if len(row) != num_orig_vars:
            return f"constraint matrix row length {len(row)} does not match number of variables {num_orig_vars}"
    try:
        A = np.array(problem.constraint_matrix, dtype=float)
        rhs = np.array(problem.constraint_rhs, dtype=float)
    except (ValueError, TypeError):
        return "constraint_matrix and constraint_rhs must be numeric"
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(rhs)):
        return "constraint_matrix and constraint_rhs must be finite (no NaN or Inf)"

    valid_senses = {"<=", ">=", "=="}
    for s in problem.constraint_senses:
        if s not in valid_senses:
            return f"invalid constraint sense {s!r}; must be one of <=, >=, =="
    return None


def solve_lp_problem(problem: LPProblemInput) -> LPSolution:
    # 1. Validation
    err = _validate(problem)
    if err is not None:
        return LPSolution(status=STATUS_ERROR, message=err)
    sense = problem.objective_sense.strip().lower()
    is_maximize = sense == "maximize"

    try:
        # 2. Standard form
        state, needs_phase_one = build_standard_form(problem, is_maximize=is_maximize)

        # 3. Phase I (only if artificials exist)
        if needs_phase_one:
            # Phase I cost: 1 on every artificial column, 0 elsewhere.
            state.tableau[COST_ROW, :] = 0.0
            for col in state.artificial_cols:
                state.tableau[COST_ROW, col] = 1.0
            phase1_result = run_simplex(state, cost_row_idx=COST_ROW, banned_cols=set())

            if phase1_result == "unbounded_direction":
                return LPSolution(
                    status=STATUS_INFEASIBLE,
                    message="phase I is unbounded; original problem is infeasible",
                )
            if phase1_result == "max_iterations":
                return LPSolution(
                    status=STATUS_ERROR,
                    message="phase I did not converge (cycling suspected)",
                )
            # Feasibility check: any artificial still basic with positive value?
            artificial_sum = 0.0
            for i in range(len(state.basis)):
                if state.basis[i] in state.artificial_cols:
                    artificial_sum += state.tableau[CONSTRAINTS_START_ROW + i, state.rhs_col]
            if artificial_sum > TOL:
                return LPSolution(
                    status=STATUS_INFEASIBLE,
                    message="original problem has no feasible solution (artificial variables remain in basis)",
                )

            # 4. Set Phase II cost row on COST_ROW (overwriting the Phase I cost).
            state.tableau[COST_ROW, :] = 0.0
            obj = np.array(problem.objective_coefficients, dtype=float)
            if is_maximize:
                obj = -obj
            state.tableau[COST_ROW, :state.num_orig_vars] = obj

        # 5. Phase II: artificials banned from re-entering.
        phase2_result = run_simplex(
            state, cost_row_idx=COST_ROW, banned_cols=state.artificial_cols
        )

        if phase2_result == "unbounded_direction":
            return LPSolution(status=STATUS_UNBOUNDED, message="problem is unbounded")
        if phase2_result == "max_iterations":
            return LPSolution(
                status=STATUS_ERROR,
                message="phase II did not converge (cycling suspected)",
            )

        # 6. Extract solution.
        sol_vars, obj_val, has_multiple = extract_solution(
            state, cost_row_idx=COST_ROW, is_maximize=is_maximize
        )
        return LPSolution(
            status=STATUS_OPTIMAL,
            solution_variables=list(sol_vars),
            objective_value=float(obj_val),
            has_multiple_optima=has_multiple,
        )
    except Exception as e:  # last-resort safety net; contract is "never raise"
        return LPSolution(status=STATUS_ERROR, message=f"internal solver error: {e}")
