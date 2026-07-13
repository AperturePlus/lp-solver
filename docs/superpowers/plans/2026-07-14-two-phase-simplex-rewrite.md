# Two-Phase Simplex Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken Big-M simplex in `lp_solver/solver/simplex.py` with a correct two-phase simplex, harden input validation, restore the `LPSolution` status contract with a `has_multiple_optima` field, fix the GUI variable-name display and status text, and add pytest regression tests — fixing all 13 defects from the code review.

**Architecture:** A new `simplex_core.py` holds pure, unit-testable tableau primitives (`TableauState` dataclass + `build_standard_form`, `select_entering`, `ratio_test`, `pivot`, `run_simplex`, `extract_solution`) that know nothing about phases. `simplex.py` orchestrates Phase I (minimize artificial sum → feasibility) then Phase II (original objective, artificials banned) and maps results to a final `LPSolution`. Bland's rule throughout guarantees termination.

**Tech Stack:** Python 3.13, NumPy (already a dependency), pytest (added as dev dependency). No scipy in production code (used only to verify expected test answers, already done).

**Reference spec:** `docs/superpowers/specs/2026-07-14-two-phase-simplex-rewrite-design.md`

## Global Constraints

- Python ≥ 3.13 (pyproject.toml already requires `>=3.13`).
- NumPy is the only numeric dependency; it is already in `requirements.txt`.
- `TOL = 1e-9` is the single tolerance constant used everywhere in the core (optimality, ratio denominator, infeasibility, multiple-optima). No `np.isclose`.
- `status` is always a bare token from the constants `STATUS_OPTIMAL`/`STATUS_UNBOUNDED`/`STATUS_INFEASIBLE`/`STATUS_ERROR` — never a decorated string.
- `solve_lp_problem` never raises; every failure path returns `LPSolution(status=STATUS_ERROR, ...)`.
- `self.custom_var_names` in `main_window.py` is a **list** (0-indexed), defaulting to `["x1", "x2", ...]` — index it directly, do not use `.get()`.
- All new tests must fail on the old code and pass on the new (acceptance criterion).

---

## File Structure

- **Create:** `lp_solver/solver/simplex_core.py` — pure tableau primitives + `TableauState` dataclass. No phase logic, no I/O, no `LPProblemInput` dependency beyond `build_standard_form`.
- **Modify:** `lp_solver/solver/simplex.py` — keep public `solve_lp_problem` + `STATUS_*` constants; replace body with two-phase orchestration calling `simplex_core`. Delete `BIG_M_VALUE` and the old Big-M body.
- **Modify:** `lp_solver/solver/models.py` — add `has_multiple_optima: Optional[bool] = None` to `LPSolution`.
- **Modify:** `lp_solver/gui/main_window.py` — result-display block (lines ~315-327): variable-name lookup + status text.
- **Modify:** `pyproject.toml` — add `[project.optional-dependencies] dev = ["pytest>=8"]`.
- **Create:** `tests/conftest.py`, `tests/test_validation.py`, `tests/test_simplex_core.py`, `tests/test_two_phase.py`.
- **Create:** `tests/__init__.py` (empty, so pytest discovery is stable on this layout).

---

## Task 1: Add `has_multiple_optima` field to `LPSolution`

**Files:**
- Modify: `lp_solver/solver/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `LPSolution(status, solution_variables=None, objective_value=None, has_multiple_optima=None, message="")` — the field order later tasks rely on.

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` (empty) and `tests/test_models.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL with `AttributeError` or dataclass rejecting unknown kwarg `has_multiple_optima` (field does not exist yet). If pytest is not installed, install it first: `pip install pytest`.

- [ ] **Step 3: Add the field**

In `lp_solver/solver/models.py`, edit the `LPSolution` dataclass so the fields read (keep existing order, insert `has_multiple_optima` before `message`):

```python
@dataclass
class LPSolution:
    """
    data structure for the solution of linear programming problems.
    """
    status: str  # solution status: "optimal", "unbounded", "infeasible", "error"
    solution_variables: Optional[List[float]] = None  # optimal solution values of variables
    objective_value: Optional[float] = None  # optimal value of the objective function
    has_multiple_optima: Optional[bool] = None  # True/False when status=="optimal", else None
    message: str = ""  # additional information, such as error messages or solution explanations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add lp_solver/solver/models.py tests/__init__.py tests/test_models.py
git commit -m "feat(models): add has_multiple_optima field to LPSolution"
```

---

## Task 2: `TableauState` dataclass + `build_standard_form`

**Files:**
- Create: `lp_solver/solver/simplex_core.py`
- Test: `tests/test_simplex_core.py`

**Interfaces:**
- Consumes: `LPProblemInput` from `lp_solver.solver.models`, `numpy as np`.
- Produces:
  - `TableauState` dataclass with fields: `tableau: np.ndarray`, `basis: list[int]`, `num_orig_vars: int`, `slack_cols: set[int]`, `surplus_cols: set[int]`, `artificial_cols: set[int]`, `rhs_col: int`.
  - `build_standard_form(problem, is_maximize) -> tuple[TableauState, bool]` returning `(state, needs_phase_one)`. `needs_phase_one` is `True` iff any artificial variable exists.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_simplex_core.py`:

```python
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
    # min problem stores -c; cost row is row 1 (C_ROW=1) by convention
    # original obj [3,2] negated -> [-3,-2] in columns 0,1
    assert state.tableau[1, 0] == pytest.approx(-3.0)
    assert state.tableau[1, 1] == pytest.approx(-2.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_simplex_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lp_solver.solver.simplex_core'`.

- [ ] **Step 3: Implement `simplex_core.py` with the dataclass and builder**

Create `lp_solver/solver/simplex_core.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_simplex_core.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add lp_solver/solver/simplex_core.py tests/test_simplex_core.py
git commit -m "feat(core): add TableauState and build_standard_form"
```

---

## Task 3: `select_entering` (Bland's rule) + `ratio_test` + `pivot`

**Files:**
- Modify: `lp_solver/solver/simplex_core.py` (append functions)
- Modify: `tests/test_simplex_core.py` (append tests)

**Interfaces:**
- Produces:
  - `select_entering(state, cost_row_idx, banned_cols) -> int | None` — lowest-index column with `tableau[cost_row_idx, j] < -TOL`, skipping `banned_cols`; `None` at optimality.
  - `ratio_test(state, pivot_col) -> int | None` — returns leaving **tableau row index** (≥ `CONSTRAINTS_START_ROW`) or `None` if unbounded direction. Bland tie-break: lowest row index on equal ratio.
  - `pivot(state, pivot_row, pivot_col, cost_row_idx) -> None` — in-place normalize + eliminate across all rows including `cost_row_idx`; updates `basis`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_simplex_core.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_simplex_core.py -v`
Expected: FAIL with `ImportError: cannot import name 'select_entering'`.

- [ ] **Step 3: Implement the three functions**

Append to `lp_solver/solver/simplex_core.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_simplex_core.py -v`
Expected: PASS (all tests so far, 11 total).

- [ ] **Step 5: Commit**

```bash
git add lp_solver/solver/simplex_core.py tests/test_simplex_core.py
git commit -m "feat(core): add select_entering (Bland), ratio_test, pivot"
```

---

## Task 4: `run_simplex` + `extract_solution`

**Files:**
- Modify: `lp_solver/solver/simplex_core.py` (append functions)
- Modify: `tests/test_simplex_core.py` (append tests)

**Interfaces:**
- Produces:
  - `run_simplex(state, cost_row_idx, banned_cols) -> str` — returns `"optimal"`, `"unbounded_direction"`, or `"max_iterations"`. Drives the cost row to canonical form before iterating (subtracts basic-var rows with nonzero cost from the cost row).
  - `extract_solution(state, cost_row_idx, is_maximize) -> tuple[np.ndarray, float, bool]` — returns `(solution_vars, objective_value, has_multiple_optima)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_simplex_core.py`:

```python
from lp_solver.solver.simplex_core import run_simplex, extract_solution


def test_run_simplex_optimal_le_problem():
    # min -x1 s.t. x1 <= 10  ->  optimal at x1=10, obj=-10
    state = _le_state()
    result = run_simplex(state, cost_row_idx=0, banned_cols=set())
    assert result == "optimal"
    sol, obj, multiple = extract_solution(state, cost_row_idx=0, is_maximize=True)
    assert sol[0] == pytest.approx(10.0)
    assert obj == pytest.approx(10.0)  # re-negated for maximize
    assert multiple is False


def test_run_simplex_returns_unbounded_direction():
    # min -x1 with no upper bound on x1 -> unbounded
    problem = LPProblemInput(
        objective_coefficients=[1.0],
        objective_sense="maximize",
        constraint_matrix=[[1.0]],
        constraint_senses=[">="],
        constraint_rhs=[0.0],
    )
    state, _ = build_standard_form(problem, is_maximize=True)
    # Phase II cost = [-1, ...]; x1 can grow unbounded
    result = run_simplex(state, cost_row_idx=0, banned_cols=state.artificial_cols)
    assert result == "unbounded_direction"


def test_extract_solution_multiple_optima_only_over_orig_vars():
    # max x1 s.t. x1<=10 with obj=[1,0] (x2 zero-cost, slack non-basic)
    # unique optimum x1=10; slack has zero reduced cost but must NOT trigger multiple
    problem = LPProblemInput(
        objective_coefficients=[1.0, 0.0],
        objective_sense="maximize",
        constraint_matrix=[[1.0, 0.0]],
        constraint_senses=["<="],
        constraint_rhs=[10.0],
    )
    state, _ = build_standard_form(problem, is_maximize=True)
    run_simplex(state, cost_row_idx=0, banned_cols=set())
    sol, obj, multiple = extract_solution(state, cost_row_idx=0, is_maximize=True)
    assert obj == pytest.approx(10.0)
    assert multiple is False  # regression: old code reported True


def test_extract_solution_detects_true_multiple_optima():
    # max x1+x2 s.t. x1+x2<=4 -> multiple optima along the edge
    problem = LPProblemInput(
        objective_coefficients=[1.0, 1.0],
        objective_sense="maximize",
        constraint_matrix=[[1.0, 1.0]],
        constraint_senses=["<="],
        constraint_rhs=[4.0],
    )
    state, _ = build_standard_form(problem, is_maximize=True)
    run_simplex(state, cost_row_idx=0, banned_cols=set())
    sol, obj, multiple = extract_solution(state, cost_row_idx=0, is_maximize=True)
    assert obj == pytest.approx(4.0)
    assert multiple is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_simplex_core.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_simplex'`.

- [ ] **Step 3: Implement `run_simplex` and `extract_solution`**

Append to `lp_solver/solver/simplex_core.py`:

```python
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


def _artificial_value_sum(state: "TableauState") -> float:
    """Sum of artificial variables currently in the basis."""
    total = 0.0
    for i in range(len(state.basis)):
        if state.basis[i] in state.artificial_cols:
            total += state.tableau[CONSTRAINTS_START_ROW + i, state.rhs_col]
    return total


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_simplex_core.py -v`
Expected: PASS (all tests, 15 total).

- [ ] **Step 5: Commit**

```bash
git add lp_solver/solver/simplex_core.py tests/test_simplex_core.py
git commit -m "feat(core): add run_simplex and extract_solution"
```

---

## Task 5: `solve_lp_problem` two-phase orchestration + validation

**Files:**
- Modify: `lp_solver/solver/simplex.py` (replace body, keep `STATUS_*` constants)
- Test: `tests/test_two_phase.py`, `tests/test_validation.py`

**Interfaces:**
- Consumes: `build_standard_form`, `run_simplex`, `extract_solution`, `TOL`, `CONSTRAINTS_START_ROW` from `simplex_core`; `LPProblemInput`/`LPSolution` from `models`.
- Produces: `solve_lp_problem(problem: LPProblemInput) -> LPSolution` (public, unchanged signature).

- [ ] **Step 1: Write the failing end-to-end tests**

Create `tests/test_two_phase.py`:

```python
import pytest
from lp_solver.solver.models import LPProblemInput, LPSolution
from lp_solver.solver.simplex import solve_lp_problem


def test_ge_constraint_min():
    # min x1+x2 s.t. x1+x2>=2 -> optimal obj=2 (old code: error/cycling)
    p = LPProblemInput([1.0, 1.0], "minimize", [[1.0, 1.0]], [">="], [2.0])
    s = solve_lp_problem(p)
    assert s.status == "optimal"
    assert s.objective_value == pytest.approx(2.0)
    assert s.has_multiple_optima is False


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
    # max x1 with obj=[1,0] s.t. x1<=10 -> unique (old code: multiple)
    p = LPProblemInput([1.0, 0.0], "maximize", [[1.0, 0.0]], ["<="], [10.0])
    s = solve_lp_problem(p)
    assert s.status == "optimal"
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
```

Create `tests/test_validation.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_two_phase.py tests/test_validation.py -v`
Expected: FAIL — old solver still in place; `test_ge_constraint_min` returns `error`, `test_negative_rhs_le_flip_is_infeasible` returns `unbounded`, etc.

- [ ] **Step 3: Replace `simplex.py` with the two-phase orchestrator**

Replace the entire contents of `lp_solver/solver/simplex.py` with:

```python
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
    if sense not in ("max", "maximize", "min", "minimize"):
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
    is_maximize = sense in ("max", "maximize")

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: PASS (all tests: models, core, two-phase, validation).

- [ ] **Step 5: Commit**

```bash
git add lp_solver/solver/simplex.py tests/test_two_phase.py tests/test_validation.py
git commit -m "feat(solver): replace Big-M with two-phase simplex + validation"
```

---

## Task 6: GUI — custom variable names + status text

**Files:**
- Modify: `lp_solver/gui/main_window.py` (result-display block, ~lines 317-327)
- Test: manual (GUI); a focused unit test on the formatting logic is optional but encouraged.

**Interfaces:**
- Consumes: `LPSolution.has_multiple_optima` (Task 1), `self.custom_var_names` (list, 0-indexed).
- Produces: updated display block.

- [ ] **Step 1: Read the current display block to confirm exact line numbers**

Open `lp_solver/gui/main_window.py` and locate the block beginning `self.results_output.clear()` (around line 317). Confirm the lines match:

```python
            self.results_output.clear()
            self.results_output.append(f"Status: {solution.status}")
            if solution.objective_value is not None:
                self.results_output.append(f"Objective Value: {solution.objective_value:.4f}")
            if solution.solution_variables is not None:
                formatted_vars = [f"x{idx+1}={var:.4f}" for idx, var in enumerate(solution.solution_variables)]
                self.results_output.append(f"Solution Variables: {', '.join(formatted_vars)}")
            else: # e.g. for unbounded or infeasible if variables are not set
                self.results_output.append(f"Solution Variables: Not applicable")
            if solution.message:
                self.results_output.append(f"Message: {solution.message}")
```

- [ ] **Step 2: Replace the block**

Replace that block with:

```python
            self.results_output.clear()
            if solution.status == "optimal":
                if solution.has_multiple_optima:
                    status_line = "状态: 最优解 (存在多重最优解)"
                else:
                    status_line = "状态: 最优解 (唯一最优解)"
            elif solution.status == "infeasible":
                status_line = "状态: 无可行解"
            elif solution.status == "unbounded":
                status_line = "状态: 无界解"
            else:  # "error"
                status_line = "状态: 求解失败"
            self.results_output.append(status_line)
            if solution.objective_value is not None:
                self.results_output.append(f"目标函数值: {solution.objective_value:.4f}")
            if solution.solution_variables is not None:
                names = list(self.custom_var_names)
                # pad/trim to solution length defensively
                while len(names) < len(solution.solution_variables):
                    names.append(f"x{len(names)+1}")
                formatted_vars = [
                    f"{names[idx]}={var:.4f}"
                    for idx, var in enumerate(solution.solution_variables)
                ]
                self.results_output.append(f"决策变量: {', '.join(formatted_vars)}")
            else:
                self.results_output.append("决策变量: 不适用")
            if solution.message:
                self.results_output.append(f"信息: {solution.message}")
```

- [ ] **Step 3: Smoke-test the GUI launches without import/syntax errors**

Run:
```bash
cd d:/pyprj/lpq/LPQ && python -c "import ast; ast.parse(open('lp_solver/gui/main_window.py', encoding='utf-8').read()); print('syntax ok')"
```
Expected: prints `syntax ok`.

- [ ] **Step 4: Re-run the full test suite to confirm no regressions**

Run: `python -m pytest tests/ -q`
Expected: PASS (GUI change does not affect solver tests).

- [ ] **Step 5: Commit**

```bash
git add lp_solver/gui/main_window.py
git commit -m "feat(gui): show custom variable names and localized status text"
```

---

## Task 7: Add pytest dev dependency + final verification

**Files:**
- Modify: `pyproject.toml`
- Test: full suite.

- [ ] **Step 1: Add the dev dependency**

In `pyproject.toml`, after the `dependencies = []` line, add:

```toml

[project.optional-dependencies]
dev = ["pytest>=8"]
```

- [ ] **Step 2: Install dev deps and run the full suite**

Run:
```bash
cd d:/pyprj/lpq/LPQ && pip install -e ".[dev]" && python -m pytest tests/ -v
```
Expected: all tests PASS (models + core + two-phase + validation).

- [ ] **Step 3: Verify the original failing inputs from the review are now fixed**

Run:
```bash
cd d:/pyprj/lpq/LPQ && python -c "
import sys; sys.path.insert(0,'lp_solver')
from solver.models import LPProblemInput
from solver.simplex import solve_lp_problem
cases = [
  ('min x1+x2 >=2', LPProblemInput([1,1],'minimize',[[1,1]],['>='],[2]), 'optimal', 2.0),
  ('max x1<=-5', LPProblemInput([1],'maximize',[[1]],['<='],[-5]), 'infeasible', None),
  ('max x1<=10', LPProblemInput([1,0],'maximize',[[1,0]],['<='],[10]), 'optimal', 10.0),
  ('sense=max', LPProblemInput([3,2],'max',[[1,1],[1,3]],['<=','<='],[4,6]), 'error', None),
]
for name,p,st,obj in cases:
    s = solve_lp_problem(p)
    print(f'{name}: status={s.status} obj={s.objective_value} multiple={s.has_multiple_optima}')
    assert s.status == st, f'{name}: expected {st}, got {s.status}'
    if obj is not None:
        assert abs(s.objective_value - obj) < 1e-6
print('ALL REVIEW REGRESSIONS FIXED')
"
```
Expected: prints four status lines and `ALL REVIEW REGRESSIONS FIXED`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pytest dev dependency"
```

---

## Self-Review Notes (completed by plan author)

- **Spec coverage:** Every spec section maps to a task — Task 1 (data model §), Task 2-4 (algorithm § incl. build/entering/ratio/pivot/run/extract), Task 5 (validation § + two-phase orchestration + status decision table + dead-code removal), Task 6 (GUI §), Task 7 (test dependency §). The traceability table's 13 findings each map: #1 Task 3 (Bland) + Task 4 (MAX_ITER); #2 Task 5 (banned_cols in Phase II); #3 Task 5 (Phase I feasibility check before Phase II); #4 Task 5 (finiteness validation); #5 Task 5 (sense normalization rejects unknown); #6 Task 5 (type check); #7 Task 4 (extract scans only orig vars); #8 Task 1 + Task 5 (bare token); #9 Task 4 (MAX_ITER); #10 Task 2-5 (test suite); #11 Task 5 (pivot helper unifies, old dead block gone); #12 Task 5 (BIG_M_VALUE deleted); #13 Task 6 (custom_var_names).
- **Placeholder scan:** No TBD/TODO. Every step has concrete code or exact commands.
- **Type consistency:** `COST_ROW=0`, `CONSTRAINTS_START_ROW=2`, `TOL=1e-9` defined once in `simplex_core.py` and imported by `simplex.py`. `TableauState` fields match across `build_standard_form` (producer) and `select_entering`/`ratio_test`/`pivot`/`run_simplex`/`extract_solution` (consumers). `run_simplex` return strings (`"optimal"`/`"unbounded_direction"`/`"max_iterations"`) match the orchestrator's branches in Task 5.
- **Correction vs. spec:** The spec's GUI Fix 1 used `self.custom_var_names.get(idx, ...)` assuming a dict; verified `custom_var_names` is a **list** (main_window.py:121,148). The plan indexes the list directly with defensive padding — corrected.
