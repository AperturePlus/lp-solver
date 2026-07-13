from lp_solver.solver.models import LPSolution


def test_has_multiple_optima_defaults_to_none():
    sol = LPSolution(status="optimal")
    assert sol.has_multiple_optima is None


def test_has_multiple_optima_can_be_set():
    sol = LPSolution(status="optimal", has_multiple_optima=True)
    assert sol.has_multiple_optima is True


def test_existing_fields_still_work():
    sol = LPSolution(status="error", message="bad input")
    assert sol.status == "error"
    assert sol.message == "bad input"
    assert sol.solution_variables is None
    assert sol.objective_value is None
