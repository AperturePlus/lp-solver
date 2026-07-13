# Pure tableau primitives for two-phase simplex. No phase logic, no I/O.
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from .models import LPProblemInput

# Single tolerance used everywhere: optimality, ratio, infeasibility, multiple-optima.
TOL = 1e-9

# Tableau row indices. Row 0 holds the cost row used by run_simplex (Phase I or II).
# Row 1 is reserved as a second cost slot so Phase II can reuse the tableau without
# rebuilding; constraint rows start at row 2.
COST_ROW = 0
RESERVED_COST_ROW = 1
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
