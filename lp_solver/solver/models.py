# data models for linear programming problems
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class LPProblemInput:
    """
    data structure for the input of linear programming problems.
    """
    objective_coefficients: List[float]  # coefficients of the objective function
    objective_sense: str               # objective type: "maximize" or "minimize"
    constraint_matrix: List[List[float]] # coefficients of the constraint matrix
    constraint_senses: List[str]         # list of constraint types: "<=", ">=", "=="
    constraint_rhs: List[float]          # RHS values of constraints
    # assume all variables are non-negative, if other variable bounds are needed, extend here

@dataclass
class LPSolution:
    """
    data structure for the solution of linear programming problems.
    """
    status: str  # solution status: "optimal", "unbounded", "infeasible", "error"
    solution_variables: Optional[List[float]] = None  # optimal solution values of variables
    objective_value: Optional[float] = None  # optimal value of the objective function
    message: str = ""  # additional information, such as error messages or solution explanations 