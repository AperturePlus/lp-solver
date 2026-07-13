import pytest
from lp_solver.solver.models import LPProblemInput, LPSolution
from lp_solver.solver.simplex import solve_lp_problem


def test_ge_constraint_min():
    # min x1+x2 s.t. x1+x2>=2 -> optimal obj=2 (old code: error/cycling)
    # The optimum is a continuum (any x1+x2=2 with x1,x2>=0), so has_multiple_optima is True.
    p = LPProblemInput([1.0, 1.0], "minimize", [[1.0, 1.0]], [">="], [2.0])
    s = solve_lp_problem(p)
    assert s.status == "optimal"
    assert s.objective_value == pytest.approx(2.0)
    assert s.has_multiple_optima is True


def test_equality_constraint():
    # min x1+x2 s.t. x1+x2==2 -> optimal obj=2 (old code: error)
    p = LPProblemInput([1.0, 1.0], "minimize", [[1.0, 1.0]], ["=="], [2.0])
    s = solve_lp_problem(p)
    assert s.status == "optimal"
    assert s.objective_value == pytest.approx(2.0)


def test_negative_rhs_le_flip_is_infeasible():
    # max x1 s.t. x1<=-5 -> infeasible (old code: unbounded)
    p = LPProblemInput([1.0], "maximize", [[1.0]], ["<="], [-5.0])
    s = solve_lp_problem(p)
    assert s.status == "infeasible"
    assert s.has_multiple_optima is None


def test_infeasible_equality():
    # x1==1, x1==2 -> infeasible
    p = LPProblemInput([1.0], "minimize", [[1.0], [1.0]], ["==", "=="], [1.0, 2.0])
    s = solve_lp_problem(p)
    assert s.status == "infeasible"


def test_unbounded_max():
    # max x1 s.t. x1-x2<=10, 2x1<=40, x1>=1  -- make it genuinely unbounded:
    # max x1 s.t. x1>=0 (trivially unbounded)
    p = LPProblemInput([1.0], "maximize", [[1.0]], [">="], [0.0])
    s = solve_lp_problem(p)
    assert s.status == "unbounded"


def test_classic_big_m_problem():
    # min 2x1+3x2 s.t. 0.5x1+0.25x2<=4, x1+3x2>=20, x1+x2==10
    p = LPProblemInput(
        [2.0, 3.0],
        "minimize",
        [[0.5, 0.25], [1.0, 3.0], [1.0, 1.0]],
        ["<=", ">=", "=="],
        [4.0, 20.0, 10.0],
    )
    s = solve_lp_problem(p)
    assert s.status == "optimal"
    assert s.objective_value == pytest.approx(25.0)
    assert s.solution_variables[0] == pytest.approx(5.0)
    assert s.solution_variables[1] == pytest.approx(5.0)


def test_pure_le_max_baseline():
    # max 3x1+5x2 s.t. x1<=4, 2x2<=12, 3x1+2x2<=18 -> obj=36 at (2,6)
    p = LPProblemInput(
        [3.0, 5.0],
        "maximize",
        [[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]],
        ["<=", "<=", "<="],
        [4.0, 12.0, 18.0],
    )
    s = solve_lp_problem(p)
    assert s.status == "optimal"
    assert s.objective_value == pytest.approx(36.0)


def test_multiple_optima_true():
    # max x1+x2 s.t. x1+x2<=4 -> multiple
    p = LPProblemInput([1.0, 1.0], "maximize", [[1.0, 1.0]], ["<="], [4.0])
    s = solve_lp_problem(p)
    assert s.status == "optimal"
    assert s.has_multiple_optima is True


def test_unique_optima_zero_cost_var():
    # Regression for review finding #13 (old code falsely reported multiple-optima
    # because slacks/surplus have zero reduced cost). Use a TRULY unique optimum
    # where slacks are non-basic with zero reduced cost: max x1+x2 s.t. x1<=4, x2<=3
    # -> unique optimum obj=7 at (4,3); both slacks non-basic, zero reduced cost.
    # (The original review case `max x1 obj=[1,0] s.t. x1<=10` has a free variable x2
    # and is genuinely multiple-optimal, so it cannot test this regression.)
    p = LPProblemInput(
        [1.0, 1.0],
        "maximize",
        [[1.0, 0.0], [0.0, 1.0]],
        ["<=", "<="],
        [4.0, 3.0],
    )
    s = solve_lp_problem(p)
    assert s.status == "optimal"
    assert s.objective_value == pytest.approx(7.0)
    assert s.has_multiple_optima is False


def test_max_objective_value():
    # max 3x1+2x2 s.t. x1+x2<=4, x1+3x2<=6 -> obj=12 (old "max" sense bug: -0.0)
    p = LPProblemInput(
        [3.0, 2.0],
        "maximize",
        [[1.0, 1.0], [1.0, 3.0]],
        ["<=", "<="],
        [4.0, 6.0],
    )
    s = solve_lp_problem(p)
    assert s.status == "optimal"
    assert s.objective_value == pytest.approx(12.0)
