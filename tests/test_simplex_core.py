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
