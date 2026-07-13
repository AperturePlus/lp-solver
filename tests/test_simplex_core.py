import numpy as np
import pytest
from lp_solver.solver.models import LPProblemInput
from lp_solver.solver.simplex_core import TableauState, build_standard_form


def test_build_standard_form_le_only_no_artificials():
    # max x1 s.t. x1 <= 10  ->  min -x1, slack only
    problem = LPProblemInput(
        objective_coefficients=[1.0],
        objective_sense="maximize",
        constraint_matrix=[[1.0]],
        constraint_senses=["<="],
        constraint_rhs=[10.0],
    )
    state, needs_phase_one = build_standard_form(problem, is_maximize=True)
    assert needs_phase_one is False
    assert state.artificial_cols == set()
    assert state.num_orig_vars == 1
    assert len(state.basis) == 1
    # slack column is basic in row 0
    assert state.basis[0] in state.slack_cols


def test_build_standard_form_ge_adds_artificial():
    # min x1+x2 s.t. x1+x2 >= 2  ->  surplus + artificial
    problem = LPProblemInput(
        objective_coefficients=[1.0, 1.0],
        objective_sense="minimize",
        constraint_matrix=[[1.0, 1.0]],
        constraint_senses=[">="],
        constraint_rhs=[2.0],
    )
    state, needs_phase_one = build_standard_form(problem, is_maximize=False)
    assert needs_phase_one is True
    assert len(state.artificial_cols) == 1
    # the artificial column is basic
    assert state.basis[0] in state.artificial_cols


def test_build_standard_form_negative_rhs_flips_sense():
    # max x1 s.t. x1 <= -5  ->  rhs negated, row negated, sense flipped to >=
    problem = LPProblemInput(
        objective_coefficients=[1.0],
        objective_sense="maximize",
        constraint_matrix=[[1.0]],
        constraint_senses=["<="],
        constraint_rhs=[-5.0],
    )
    state, needs_phase_one = build_standard_form(problem, is_maximize=True)
    # After negation: -x1 >= 5, so an artificial is needed
    assert needs_phase_one is True
    # RHS in the tableau is now positive 5
    assert state.tableau[2, state.rhs_col] == pytest.approx(5.0)


def test_build_standard_form_maximizes_negates_objective():
    problem = LPProblemInput(
        objective_coefficients=[3.0, 2.0],
        objective_sense="maximize",
        constraint_matrix=[[1.0, 1.0]],
        constraint_senses=["<="],
        constraint_rhs=[4.0],
    )
    state, _ = build_standard_form(problem, is_maximize=True)
    # min problem stores -c; cost row is row 0 (COST_ROW=0) by convention
    # original obj [3,2] negated -> [-3,-2] in columns 0,1
    assert state.tableau[0, 0] == pytest.approx(-3.0)
    assert state.tableau[0, 1] == pytest.approx(-2.0)


from lp_solver.solver.simplex_core import (
    select_entering,
    ratio_test,
    pivot,
    CONSTRAINTS_START_ROW,
)


def _le_state():
    # min -x1 s.t. x1 <= 10; cost row already = [-1, 0(slack), 0]
    problem = LPProblemInput(
        objective_coefficients=[1.0],
        objective_sense="maximize",
        constraint_matrix=[[1.0]],
        constraint_senses=["<="],
        constraint_rhs=[10.0],
    )
    state, _ = build_standard_form(problem, is_maximize=True)
    return state


def test_select_entering_returns_lowest_index_negative():
    state = _le_state()
    # cost row = [-1, 0, 0]; column 0 is the only negative -> entering
    assert select_entering(state, cost_row_idx=0, banned_cols=set()) == 0


def test_select_entering_returns_none_at_optimality():
    state = _le_state()
    # set cost row to all non-negative
    state.tableau[0, :] = 0.0
    assert select_entering(state, cost_row_idx=0, banned_cols=set()) is None


def test_select_entering_skips_banned_cols():
    state = _le_state()
    # ban column 0; no other negative -> None
    assert select_entering(state, cost_row_idx=0, banned_cols={0}) is None


def test_ratio_test_picks_smallest_ratio():
    state = _le_state()
    # constraint row 2: [1, 1(slack), 10]; ratio = 10/1 = 10 -> leaving row 2
    leaving = ratio_test(state, pivot_col=0)
    assert leaving == CONSTRAINTS_START_ROW


def test_ratio_test_returns_none_when_unbounded():
    # build a state where pivot_col has no positive entry in any constraint row
    state = _le_state()
    state.tableau[CONSTRAINTS_START_ROW, 0] = -1.0  # now negative
    assert ratio_test(state, pivot_col=0) is None


def test_ratio_test_bland_tiebreak_lowest_index():
    # two rows with equal ratio -> lowest index wins
    problem = LPProblemInput(
        objective_coefficients=[1.0, 1.0],
        objective_sense="minimize",
        constraint_matrix=[[1.0, 0.0], [1.0, 0.0]],
        constraint_senses=["<=", "<="],
        constraint_rhs=[2.0, 2.0],
    )
    state, _ = build_standard_form(problem, is_maximize=False)
    # pivot_col 0: row2 ratio 2/1, row3 ratio 2/1 -> tie, pick row2
    assert ratio_test(state, pivot_col=0) == CONSTRAINTS_START_ROW


def test_pivot_normalizes_and_updates_basis():
    state = _le_state()
    pivot(state, pivot_row=CONSTRAINTS_START_ROW, pivot_col=0, cost_row_idx=0)
    # pivot column should be 1 in pivot row, 0 elsewhere (incl cost row)
    assert state.tableau[CONSTRAINTS_START_ROW, 0] == pytest.approx(1.0)
    assert state.tableau[0, 0] == pytest.approx(0.0)
    # basis updated: row 2 now has column 0
    assert state.basis[0] == 0
