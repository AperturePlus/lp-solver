import math
import pytest
from lp_solver.solver.models import LPProblemInput
from lp_solver.solver.simplex import solve_lp_problem


def test_objective_sense_max_rejected():
    p = LPProblemInput([3.0, 2.0], "max", [[1.0, 1.0]], ["<="], [4.0])
    s = solve_lp_problem(p)
    assert s.status == "error"
    assert "objective_sense" in s.message


def test_objective_sense_none_does_not_crash():
    p = LPProblemInput([1.0], None, [[1.0]], ["<="], [5.0])  # type: ignore[arg-type]
    s = solve_lp_problem(p)
    assert s.status == "error"
    # must NOT raise AttributeError
    assert isinstance(s, object)


def test_objective_sense_case_insensitive():
    p = LPProblemInput([1.0, 1.0], "MAXIMIZE", [[1.0, 1.0]], ["<="], [4.0])
    s = solve_lp_problem(p)
    assert s.status == "optimal"


def test_nan_rhs_rejected():
    p = LPProblemInput([1.0], "maximize", [[1.0]], ["<="], [float("nan")])
    s = solve_lp_problem(p)
    assert s.status == "error"


def test_inf_objective_rejected():
    p = LPProblemInput([float("inf")], "maximize", [[1.0]], ["<="], [5.0])
    s = solve_lp_problem(p)
    assert s.status == "error"


def test_ragged_constraint_matrix_rejected():
    p = LPProblemInput([1.0, 1.0], "maximize", [[1.0]], ["<="], [5.0])
    s = solve_lp_problem(p)
    assert s.status == "error"


def test_unknown_sense_rejected():
    p = LPProblemInput([1.0], "maximize", [[1.0]], [">>"], [5.0])
    s = solve_lp_problem(p)
    assert s.status == "error"


def test_empty_objective_rejected():
    p = LPProblemInput([], "maximize", [[1.0]], ["<="], [5.0])  # type: ignore[list-item]
    s = solve_lp_problem(p)
    assert s.status == "error"
