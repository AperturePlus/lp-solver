# Pure tableau primitives for two-phase simplex. No phase logic, no I/O.
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from .models import LPProblemInput

# Single tolerance used everywhere: optimality, ratio, infeasibility, multiple-optima.
TOL = 1e-9

# Tableau row indices. Row 0 holds the cost row used by run_simplex (Phase I or II).
# Constraint rows start at row 2.
COST_ROW = 0
CONSTRAINTS_START_ROW = 2


@dataclass
class TableauState:
    tableau: np.ndarray
    basis: list  # list[int]: column index of basic var per constraint row
    num_orig_vars: int
    slack_cols: set = field(default_factory=set)
    surplus_cols: set = field(default_factory=set)
    artificial_cols: set = field(default_factory=set)
    rhs_col: int = 0  # set in build_standard_form


def build_standard_form(problem: LPProblemInput, is_maximize: bool) -> Tuple["TableauState", bool]:
    """Build the standard-form tableau. Returns (state, needs_phase_one)."""
    num_orig_vars = len(problem.objective_coefficients)
    num_constraints = len(problem.constraint_matrix)

    senses = list(problem.constraint_senses)
    rhs = np.array(problem.constraint_rhs, dtype=float)
    A = np.array(problem.constraint_matrix, dtype=float)

    # Ensure RHS non-negative: negate row + flip sense.
    for i in range(num_constraints):
        if rhs[i] < 0:
            rhs[i] *= -1
            A[i, :] *= -1
            if senses[i] == "<=":
                senses[i] = ">="
            elif senses[i] == ">=":
                senses[i] = "<="
            # '==' unchanged

    num_slack = sum(1 for s in senses if s == "<=")
    num_surplus = sum(1 for s in senses if s == ">=")
    num_artificial = sum(1 for s in senses if s == ">=" or s == "==")
    total_vars = num_orig_vars + num_slack + num_surplus + num_artificial

    # Rows: cost row, reserved cost row, constraint rows. Cols: vars + RHS.
    tableau = np.zeros((CONSTRAINTS_START_ROW + num_constraints, total_vars + 1))
    rhs_col = total_vars

    # Objective (min form): negate for maximize.
    obj = np.array(problem.objective_coefficients, dtype=float)
    if is_maximize:
        obj = -obj
    tableau[COST_ROW, :num_orig_vars] = obj  # Phase II cost; Phase I overwrites COST_ROW temporarily.

    slack_cols, surplus_cols, artificial_cols = set(), set(), set()
    basis = [-1] * num_constraints

    cur_slack = num_orig_vars
    cur_surplus = num_orig_vars + num_slack
    cur_artificial = num_orig_vars + num_slack + num_surplus

    for i in range(num_constraints):
        row = CONSTRAINTS_START_ROW + i
        tableau[row, :num_orig_vars] = A[i, :]
        tableau[row, rhs_col] = rhs[i]
        if senses[i] == "<=":
            tableau[row, cur_slack] = 1.0
            slack_cols.add(cur_slack)
            basis[i] = cur_slack
            cur_slack += 1
        elif senses[i] == ">=":
            tableau[row, cur_surplus] = -1.0
            surplus_cols.add(cur_surplus)
            cur_surplus += 1
            tableau[row, cur_artificial] = 1.0
            artificial_cols.add(cur_artificial)
            basis[i] = cur_artificial
            cur_artificial += 1
        elif senses[i] == "==":
            tableau[row, cur_artificial] = 1.0
            artificial_cols.add(cur_artificial)
            basis[i] = cur_artificial
            cur_artificial += 1

    state = TableauState(
        tableau=tableau,
        basis=basis,
        num_orig_vars=num_orig_vars,
        slack_cols=slack_cols,
        surplus_cols=surplus_cols,
        artificial_cols=artificial_cols,
        rhs_col=rhs_col,
    )
    return state, len(artificial_cols) > 0


from typing import Optional


def select_entering(state: "TableauState", cost_row_idx: int, banned_cols: set) -> Optional[int]:
    """Bland's rule: return lowest-index column with reduced cost < -TOL, not banned."""
    total_vars = state.tableau.shape[1] - 1  # exclude RHS column
    for j in range(total_vars):
        if j in banned_cols:
            continue
        if state.tableau[cost_row_idx, j] < -TOL:
            return j
    return None


def ratio_test(state: "TableauState", pivot_col: int) -> Optional[int]:
    """Minimum-ratio test with Bland tie-break (lowest row index). Returns tableau row index or None."""
    min_ratio = float("inf")
    pivot_row = None
    num_constraints = len(state.basis)
    for i in range(num_constraints):
        row = CONSTRAINTS_START_ROW + i
        coeff = state.tableau[row, pivot_col]
        if coeff > TOL:
            ratio = state.tableau[row, state.rhs_col] / coeff
            if ratio < min_ratio - TOL:
                min_ratio = ratio
                pivot_row = row
            # Bland tie-break: on a tie (within TOL) keep the FIRST (lowest index) row,
            # which is already the behavior since we only update on strict improvement.
    return pivot_row


def pivot(state: "TableauState", pivot_row: int, pivot_col: int, cost_row_idx: int) -> None:
    """In-place pivot: normalize pivot row, eliminate pivot col from all other rows, update basis."""
    pivot_element = state.tableau[pivot_row, pivot_col]
    state.tableau[pivot_row, :] /= pivot_element
    num_rows = state.tableau.shape[0]
    for i in range(num_rows):
        if i == pivot_row:
            continue
        factor = state.tableau[i, pivot_col]
        if factor != 0.0:
            state.tableau[i, :] -= factor * state.tableau[pivot_row, :]
    state.basis[pivot_row - CONSTRAINTS_START_ROW] = pivot_col


def _drive_cost_to_canonical(state: "TableauState", cost_row_idx: int) -> None:
    """Subtract basic-var rows with nonzero cost from the cost row (reduced-cost setup)."""
    num_constraints = len(state.basis)
    for i in range(num_constraints):
        row = CONSTRAINTS_START_ROW + i
        basic_col = state.basis[i]
        coeff = state.tableau[cost_row_idx, basic_col]
        if abs(coeff) > TOL:
            state.tableau[cost_row_idx, :] -= coeff * state.tableau[row, :]


def run_simplex(state: "TableauState", cost_row_idx: int, banned_cols: set) -> str:
    """Run simplex to optimality on the given cost row. Returns 'optimal', 'unbounded_direction', or 'max_iterations'."""
    _drive_cost_to_canonical(state, cost_row_idx)
    num_constraints = len(state.basis)
    total_vars = state.tableau.shape[1] - 1
    max_iter = max(1000, 50 * (state.num_orig_vars + num_constraints))
    for _ in range(max_iter):
        entering = select_entering(state, cost_row_idx, banned_cols)
        if entering is None:
            return "optimal"
        leaving = ratio_test(state, pivot_col=entering)
        if leaving is None:
            return "unbounded_direction"
        pivot(state, pivot_row=leaving, pivot_col=entering, cost_row_idx=cost_row_idx)
    return "max_iterations"


def extract_solution(state: "TableauState", cost_row_idx: int, is_maximize: bool):
    """Return (solution_vars, objective_value, has_multiple_optima)."""
    sol = np.zeros(state.num_orig_vars)
    for i in range(len(state.basis)):
        col = state.basis[i]
        if col < state.num_orig_vars:
            sol[col] = state.tableau[CONSTRAINTS_START_ROW + i, state.rhs_col]
    obj_val = -state.tableau[cost_row_idx, state.rhs_col]
    if is_maximize:
        obj_val = -obj_val
    # Multiple optima: a NON-BASIC ORIGINAL variable with ~0 reduced cost.
    has_multiple = False
    basic_set = set(state.basis)
    for j in range(state.num_orig_vars):
        if j in basic_set:
            continue
        if abs(state.tableau[cost_row_idx, j]) < TOL:
            has_multiple = True
            break
    return sol, obj_val, has_multiple
