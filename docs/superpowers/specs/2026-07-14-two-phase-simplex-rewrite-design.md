# LPQ Solver Rewrite — Two-Phase Simplex & Robustness

**Date:** 2026-07-14
**Status:** Approved (brainstorming complete)
**Scope:** Fix all 13 defects found in the code review of `lp_solver/solver/simplex.py`, by replacing the broken Big-M implementation with a two-phase simplex method, hardening input validation, tightening the `LPSolution` contract, and adding regression tests.

## Background

A max-effort code review found that the Big-M simplex in `simplex.py` is essentially unusable: every linear program containing a `>=` or `==` constraint (the core use case of a Big-M solver) fails to converge and returns `status="error"`. Additional defects include infeasible problems misreported as unbounded, `objective_sense="max"` silently treated as minimize, `objective_sense=None` crashing with `AttributeError`, NaN inputs causing false-unbounded results, a false "multiple optimal solutions" verdict, and a violated status-string contract. Every defect was confirmed by executing the actual solver on concrete inputs, not by static reading alone.

## Goals

1. Make the solver correct on all standard LP classes: `<=`/`>=`/`==` constraints, feasible / infeasible / unbounded / multiple-optima, minimize and maximize.
2. Make the solver robust: never raise, always return an `LPSolution`; reject bad input at the boundary.
3. Restore a clean `LPSolution` contract (bare status token + structured multiple-optima flag).
4. Add regression tests so the reviewed bugs cannot recur.
5. Minimal, targeted GUI fixes (variable names, status text).

## Non-Goals

- No new GUI features, widgets, or input-parsing changes.
- No replacement of the hand-written simplex with `scipy.optimize.linprog`; the project's purpose is a teaching implementation of the simplex method.
- No refactoring of unrelated code.
- No CLI/`main.py` changes.

## Architecture & Module Layout

`solve_lp_problem` remains the single public entry point (GUI contract unchanged). It orchestrates two-phase simplex via small, testable helpers. All code lives in `lp_solver/solver/`.

```
lp_solver/solver/
  models.py        # LPSolution gains has_multiple_optima field
  simplex.py       # public solve_lp_problem + two-phase orchestration
  simplex_core.py  # NEW: pure, unit-testable tableau primitives
```

### `simplex_core.py` — stateless primitives

A small `TableauState` dataclass holds the tableau array plus bookkeeping:

```python
@dataclass
class TableauState:
    tableau: np.ndarray            # (CONSTRAINTS_START_ROW + num_constraints, total_vars + 1)
    basis: list[int]               # basis column index per constraint row
    num_orig_vars: int
    slack_cols: set[int]
    surplus_cols: set[int]
    artificial_cols: set[int]
    rhs_col: int                   # == total_vars (last column)
```

Primitives (each pure, operating on `TableauState`):

- `build_standard_form(problem, is_maximize) -> TableauState` — builds the standard-form tableau: max converted to min, negative-RHS rows negated with sense flip, slack/surplus/artificial columns placed, initial basis chosen. Returns the state plus a flag `needs_phase_one` (True iff any artificial exists).
- `select_entering(state, cost_row, banned_cols) -> int | None` — **Bland's rule**: returns the lowest-index column with reduced cost `< -TOL`, skipping `banned_cols`. Returns `None` at optimality.
- `ratio_test(state, pivot_col) -> int | None` — minimum-ratio test with Bland tie-break (lowest row index on equal ratios). Returns the leaving row index, or `None` if no row has a positive pivot-column entry (unbounded direction).
- `pivot(state, pivot_row, pivot_col, cost_row_idx) -> None` — in-place: normalize pivot row, eliminate pivot column from **all** other rows (constraint rows + the cost row at `cost_row_idx`), update `basis`. The cost row lives inside the same tableau at a designated row index (e.g. row 0), so `pivot` updates it uniformly rather than the caller doing a separate update.
- `run_simplex(state, cost_row_idx, banned_cols) -> str` — the core loop. Returns one of `"optimal"`, `"unbounded_direction"`, `"max_iterations"`. Calls `select_entering` / `ratio_test` / `pivot` repeatedly. Does **not** decide final problem status — the orchestrator does.
- `extract_solution(state, cost_row, is_maximize) -> tuple[np.ndarray, float, bool]` — returns `(solution_vars, objective_value, has_multiple_optima)`. Multiple-optima detection scans **only** `range(num_orig_vars)` (original variable columns), not slack/surplus/artificial columns.

The core has **zero awareness of phases** — it is "run simplex with this cost row and this ban list." Phase semantics live entirely in `simplex.py`.

### `simplex.py` — orchestration

```python
def solve_lp_problem(problem: LPProblemInput) -> LPSolution:
    # 1. Validate (Section "Input Validation"). Returns LPSolution(status=ERROR) on any failure.
    # 2. Build standard form. If needs_phase_one: run Phase I, then feasibility check.
    # 3. Run Phase II on original cost row with artificial_cols as banned.
    # 4. Map run_simplex result + phase context to final status (Section "Status Decision").
    # 5. extract_solution, build LPSolution.
```

## Two-Phase Algorithm

### Standard-form construction (`build_standard_form`)

1. Convert to minimization: if `is_maximize`, negate `obj_coeffs_orig`.
2. For each constraint, if `rhs[i] < 0`: negate the row and flip `<=` ↔ `>=` (existing logic, kept). `==` unchanged.
3. Tableau columns: `[original vars | slack vars | surplus vars | artificial vars | RHS]`.
4. Initial basis: `<=` → slack; `>=` → surplus(-1) + artificial(+1); `==` → artificial(+1).
5. `artificial_cols` stored as a `set` for O(1) membership.

If no artificial variables are needed (all `<=`, all RHS ≥ 0), Phase I is skipped — Phase II runs directly.

### Phase I (only if artificials exist)

- Cost row `phase1_cost` lives at a designated tableau row index (e.g. row 0); values: 1 on every artificial column, 0 elsewhere.
- Drive to canonical form: subtract each artificial's constraint row from `phase1_cost` (the one piece of the old M_ROW setup that was correct).
- `run_simplex(state, phase1_cost_row_idx, banned_cols=set())`. Artificials may stay in basis during Phase I.
- **Bland's rule throughout** (lowest-index entering column) — eliminates the cycling that broke the old solver.
- **Degenerate-artificial handling:** after Phase I terminates, if an artificial remains basic at value ≈ 0, attempt to pivot it out to a non-artificial column (standard "artificial leaving" procedure). If its row is all-zero across non-artificial columns, the constraint is redundant — drop the row. This prevents degenerate artificials from interfering with Phase II.
- **Feasibility check:** after Phase I, if `sum(artificial values in basis) > TOL` → return `STATUS_INFEASIBLE` with message. This is the **only** infeasibility decision point; it happens before any Phase II unbounded declaration, fixing the false-unbounded bug.

### Phase II

- Cost row `phase2_cost`: the (minimization) original objective coefficients on original-var columns, 0 elsewhere.
- Drive to canonical form: for each basic variable with nonzero original cost, subtract its constraint row from `phase2_cost`.
- `run_simplex(state, phase2_cost, banned_cols=artificial_cols)`. **Artificial columns are banned from re-entering** — fixes the re-entry cycling bug (review finding at old L124).
- `banned_cols` checked inside `select_entering` before considering a column.

### Status decision (orchestrator)

`run_simplex` returns a phase-local result; the orchestrator maps it to a final `LPSolution.status`:

| Phase I result | Artificial sum | Final status |
|---|---|---|
| `optimal` | `> TOL` | `infeasible` |
| `optimal` | `≤ TOL` | proceed to Phase II |
| `unbounded_direction` | (any) | `infeasible` (Phase I unbounded ⇒ infeasible) |
| `max_iterations` | (any) | `error` (cycling suspected — should not happen with Bland) |

| Phase II result | Final status |
|---|---|
| `optimal` | `optimal` (+ `has_multiple_optima` from extract) |
| `unbounded_direction` | `unbounded` (artificials are out by construction here) |
| `max_iterations` | `error` |

This table is the fix for the false-unbounded bug: unboundedness in Phase I ⇒ infeasible; unboundedness in Phase II ⇒ genuinely unbounded. The old code conflated them.

### Solution extraction (`extract_solution`)

- Read original-variable values from the basis.
- `objective_value = -cost_row[RHS]` (sign verified correct during review), re-negated for maximize.
- **Multiple-optima detection fixed:** scan only `range(num_orig_vars)`. A non-basic original variable with `abs(reduced_cost) < TOL` ⇒ `has_multiple_optima=True`. Slack/surplus/artificial columns are excluded — they legitimately have zero reduced cost and caused the old false positive.

### Tolerance unification

Define `TOL = 1e-9` as a module constant in `simplex_core.py`, used everywhere: optimality (`reduced_cost > -TOL`), ratio denominator (`> TOL`), infeasibility (`artificial_sum > TOL`), multiple-optima (`abs(reduced_cost) < TOL`). No `np.isclose` in the core — the old `1e-9` vs `np.isclose` (rtol=1e-5, atol=1e-8) mismatch caused contradictory classifications.

### Iteration ceiling

`MAX_ITER = max(1000, 50 * (num_orig_vars + num_constraints))`. With Bland's rule this should never be hit; it is a safety net returning `STATUS_ERROR` with a cycling message rather than hanging. Replaces the broken `2*num_constraints + num_orig_vars` heuristic (old L109).

## Input Validation & Error Handling

All validation at the top of `solve_lp_problem`, before tableau construction. Every rejection returns `LPSolution(status=STATUS_ERROR, message=...)` — never raises.

1. **Type sanity** — `if not isinstance(problem.objective_sense, str): return error("objective_sense must be a string")`. Fixes the `None`→`AttributeError` crash.
2. **`objective_sense` normalization** — `s = problem.objective_sense.strip().lower()`. If `s in ("max", "maximize")` → maximize; elif `s in ("min", "minimize")` → minimize; **else → return error** ("unknown objective_sense: ..."). Fixes the silent-minimize bug where `"max"` was treated as minimize. No silent fallback.
3. **Dimension consistency** — `objective_coefficients` non-empty; each `constraint_matrix` row length == `num_orig_vars`; `len(matrix) == len(rhs) == len(senses)`; each sense ∈ `{"<=", ">=", "=="}`.
4. **Finiteness** — reject any NaN/Inf in `objective_coefficients`, `constraint_matrix`, `constraint_rhs` via `np.isfinite`. Fixes the NaN→false-unbounded bug at the boundary.
5. **Empty/degenerate** — `num_orig_vars == 0` → error "problem has no variables"; zero constraints → error "problem has no constraints". Accurate messages replace the generic "input data format error".

### Error message contract

Every error message is a single concrete line, e.g. `"objective_sense must be 'maximize' or 'minimize', got: 'max'"`. `status` is always `"error"` on these paths; `solution_variables`/`objective_value`/`has_multiple_optima` all `None`.

### Exception safety

`solve_lp_problem` wraps the solving body in `try/except Exception` returning `LPSolution(status=STATUS_ERROR, message=f"internal solver error: {e}")` as a last resort. The contract is "always return an LPSolution, never raise."

### Dead code removal

- Delete `BIG_M_VALUE = 1e7` (old L16) — unused; two-phase has no Big-M constant.
- Delete the redundant objective-row re-pivot (old L191-194) — the unified `pivot()` helper handles all rows once; no double-application.

## Data Model (`models.py`)

```python
@dataclass
class LPSolution:
    status: str  # one of: "optimal", "unbounded", "infeasible", "error" (bare token)
    solution_variables: Optional[List[float]] = None
    objective_value: Optional[float] = None
    has_multiple_optima: Optional[bool] = None   # NEW: True/False only when status=="optimal", else None
    message: str = ""
```

**Status contract (enforced):** `status` is strictly one of the four bare constants — never `"optimal (multiple...)"` again. Multiple-optima information moves to the structured `has_multiple_optima` boolean. `message` carries human-readable detail only.

**Field rules:**
- `status == "optimal"` → `solution_variables` and `objective_value` set; `has_multiple_optima` is `True` or `False` (never `None`).
- `status == "unbounded"` / `"infeasible"` → `solution_variables=None`, `objective_value=None`, `has_multiple_optima=None`, `message` explains.
- `status == "error"` → all optional fields `None`, `message` explains the validation failure.

`has_multiple_optima` is `Optional[bool]` (tri-state) so consumers distinguish "definitely unique" / "definitely multiple" / "not applicable".

`LPProblemInput` unchanged. `STATUS_*` constants in `simplex.py` unchanged.

## GUI Changes (`main_window.py`)

Two targeted fixes in the result-display block (L315-327). No other UI logic touched.

### Fix 1 — Custom variable names

Old L322 hardcodes `x{idx+1}`. New code reads `self.custom_var_names` (the header-rename dict, populated at L70-86), falling back to `x{idx+1}` when unset:

```python
names = [self.custom_var_names.get(idx, f"x{idx+1}")
         for idx in range(len(solution.solution_variables))]
formatted_vars = [f"{name}={var:.4f}" for name, var in zip(names, solution.solution_variables)]
```

`custom_var_names` is assumed keyed by 0-based index — to be verified against L70-86 during implementation; if keyed differently, adapt the lookup (not the data structure).

### Fix 2 — Status-aware result text

Replace the verbatim `solution.status` dump with a status-aware Chinese label:

```python
if solution.status == "optimal":
    line = "最优解 (存在多重最优解)" if solution.has_multiple_optima else "最优解 (唯一最优解)"
elif solution.status == "infeasible":
    line = "无可行解"
elif solution.status == "unbounded":
    line = "无界解"
else:  # "error"
    line = "求解失败"
self.results_output.append(f"状态: {line}")
```

Objective-value and variables lines (L319-323) kept — already correctly None-guarded. `message` still appended at the end. The try/except wrappers (L329-332) kept as defense in depth (solver should never raise now).

`lp_solver/main.py` unchanged (launches Qt window only; no solver call).

## Test Plan (`tests/`, pytest)

Rebuild the deleted `tests/` directory. Add `pytest` as a dev dependency.

```
tests/
  conftest.py           # shared LP fixtures
  test_validation.py    # input validation
  test_simplex_core.py  # unit tests for core primitives
  test_two_phase.py     # end-to-end solver tests by problem class
```

### `pyproject.toml`

```toml
[project.optional-dependencies]
dev = ["pytest>=8"]
```

Run: `python -m pytest tests/ -q`.

### `test_simplex_core.py` — unit tests

- `build_standard_form`: RHS negation flips senses; artificial columns placed correctly; `<=`-only problem produces zero artificials (`needs_phase_one == False`).
- `select_entering` (Bland): returns lowest-index negative-reduced-cost column; skips `banned_cols`; returns `None` at optimality.
- `ratio_test`: correct leaving row on a hand-computed tableau; `None` when all entries ≤ 0; Bland tie-break picks lowest index on equal ratios.
- `pivot`: normalizes pivot row, zeroes pivot column in all other rows, updates basis.
- `extract_solution`: reads original-var values; objective sign correct for min and max; multiple-optima detected only over original-var columns (regression for old L148).

### `test_two_phase.py` — end-to-end

Each asserts `status`, `objective_value`, and (where relevant) `has_multiple_optima`. Covers the exact failing inputs from the review:

| Test | Input | Expected |
|---|---|---|
| `test_ge_constraint_min` | `min x1+x2 s.t. x1+x2>=2` | optimal, obj=2 (was: error/cycling) |
| `test_equality_constraint` | `min x1+x2 s.t. x1+x2==2` | optimal, obj=2 (was: error) |
| `test_negative_rhs_le_flip` | `max x1 s.t. x1<=-5` | infeasible (was: unbounded) |
| `test_infeasible_equality` | `x1==1, x1==2` | infeasible |
| `test_unbounded_max` | standard unbounded-direction LP | unbounded |
| `test_classic_big_m_problem` | `min 2x1+3x2 s.t. 0.5x1+0.25x2<=4, x1+3x2>=20, x1+x2==10` | optimal, obj=25, vars [5,5] (expected value to be cross-checked with scipy during implementation) |
| `test_pure_le_max` | `max 3x1+5x2` textbook (`<=` only) | optimal, obj=10 (baseline regression) |
| `test_multiple_optima_true` | `max x1+x2 s.t. x1+x2<=4` | optimal, `has_multiple_optima=True` |
| `test_unique_optima_zero_cost_var` | `max x1` with `obj=[1,0]` (`<=` only) | optimal, `has_multiple_optima=False` (was: True) |
| `test_max_objective_value` | `max 3x1+2x2 s.t. x1+x2<=4, x1+3x2<=6` | optimal, obj=12 (was: -0.0 via "max" sense bug) |

### `test_validation.py`

- `objective_sense="max"` → error (not silent minimize)
- `objective_sense=None` → error (not AttributeError)
- `objective_sense="MAXIMIZE"` → treated as maximize (case-insensitive)
- NaN/Inf in rhs/objective/matrix → error
- Ragged constraint matrix → error
- Unknown sense (`">>"`) → error
- Empty objective / empty matrix → error with specific message

### `conftest.py`

Fixtures: `classic_max_problem`, `infeasible_problem`, `unbounded_problem` — each a plain `LPProblemInput` plus expected `LPSolution` fields.

### Acceptance criterion

Every bug from the review has a dedicated test that would fail on the old code and pass on the new. If a test passes on the old code, it is not a real regression test.

## Traceability — Review Findings → Fixes

| # | Review finding | Fixed by |
|---|---|---|
| 1 | No anti-cycling rule → cycling on `>=`/`==` (old L176) | Bland's rule in `select_entering` + `ratio_test`; raised `MAX_ITER` |
| 2 | Artificial re-entry via C_ROW scan (old L124) | `banned_cols=artificial_cols` in Phase II |
| 3 | Unbounded fires while artificial basic (old L179) | Phase I feasibility check before Phase II; status-decision table |
| 4 | NaN rhs → false unbounded (old L172) | Finiteness validation rejects NaN/Inf |
| 5 | `objective_sense="max"` silent minimize (old L62) | Normalization rejects unknown senses |
| 6 | `objective_sense=None` → AttributeError (old L62) | Type check returns `STATUS_ERROR` |
| 7 | False "multiple optimal solutions" on slacks (old L148) | `extract_solution` scans only original-var columns |
| 8 | Status string violates contract (old L158) | Bare token `status` + `has_multiple_optima` field |
| 9 | `max_iterations` too small (old L109) | `MAX_ITER = max(1000, 50*(n+m))` + Bland |
| 10 | Removed self-test block → no automated exercise | New pytest suite (`tests/`) |
| 11 | Redundant objective re-pivot (old L191-194) | Unified `pivot()` helper; dead block deleted |
| 12 | `BIG_M_VALUE` dead constant (old L16) | Deleted |
| 13 | GUI hardcodes `x{idx+1}` (main_window L322) | Reads `self.custom_var_names` with fallback |
