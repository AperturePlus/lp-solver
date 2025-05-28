# 主窗口 GUI 代码 
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QScrollArea, QInputDialog
)
from PyQt5.QtCore import Qt
from solver.models import LPProblemInput, LPSolution
from solver.simplex import solve_lp_problem

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linear Programming Solver")
        self.setGeometry(100, 100, 900, 700) # Adjusted size back slightly

        self.num_decision_vars = 2 # Default number of variables
        self.custom_var_names = [f"x{i+1}" for i in range(self.num_decision_vars)] # Initialize with defaults

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self._create_input_widgets()
        self._create_output_widgets()
        self._create_control_buttons()
        
        self._update_num_vars_and_constraint_table(force_default=True) # Initial setup with default vars

    def _create_input_widgets(self):
        input_group_widget = QWidget() # Use a QWidget for the input group
        input_group_layout = QVBoxLayout(input_group_widget) # Layout for the input group widget

        # --- Objective Function ---
        obj_section_layout = QHBoxLayout()
        obj_section_layout.addWidget(QLabel("Objective:"))
        self.obj_sense_combo = QComboBox()
        self.obj_sense_combo.addItems(["maximize", "minimize"])
        obj_section_layout.addWidget(self.obj_sense_combo)
        
        self.obj_coeffs_input = QLineEdit()
        self.obj_coeffs_input.setPlaceholderText("Coefficients (e.g., 3,2,5)")
        self.obj_coeffs_input.editingFinished.connect(self._update_num_vars_and_constraint_table) # Connect signal
        obj_section_layout.addWidget(self.obj_coeffs_input)
        input_group_layout.addLayout(obj_section_layout)

        # --- Constraints ---
        input_group_layout.addWidget(QLabel("Constraints (double-click variable headers like x1, x2 to rename):"))
        self.constraints_table = QTableWidget()
        self.constraints_table.horizontalHeader().sectionDoubleClicked.connect(self._handle_header_double_clicked)
        input_group_layout.addWidget(self.constraints_table)

        constraints_buttons_layout = QHBoxLayout()
        self.add_constraint_button = QPushButton("Add Constraint")
        self.add_constraint_button.clicked.connect(self.add_constraint_row)
        self.remove_constraint_button = QPushButton("Remove Selected/Last Constraint")
        self.remove_constraint_button.clicked.connect(self.remove_constraint_row)
        constraints_buttons_layout.addWidget(self.add_constraint_button)
        constraints_buttons_layout.addWidget(self.remove_constraint_button)
        input_group_layout.addLayout(constraints_buttons_layout)
        
        # Use a QScrollArea for the input group if it becomes too large
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(input_group_widget)
        
        self.layout.addWidget(scroll_area)

    def _handle_header_double_clicked(self, logical_index):
        if 0 <= logical_index < self.num_decision_vars:
            current_name = self.custom_var_names[logical_index]
            
            text, ok = QInputDialog.getText(self, "Edit Variable Name", 
                                            f"Enter new name for '{current_name}':         ", 
                                            QLineEdit.Normal, current_name)
            if ok and text.strip():
                new_name = text.strip()
                if "," in new_name or " " in new_name: # Prohibit commas and spaces
                    QMessageBox.warning(self, "Invalid Name", "Variable name cannot contain commas or spaces.")
                    return
                
                self.custom_var_names[logical_index] = new_name
                self._setup_constraints_table_columns() # Refresh headers
            elif ok and not text.strip():
                 QMessageBox.warning(self, "Invalid Name", "Variable name cannot be empty.")

    def _setup_constraints_table_columns(self):
        # self.custom_var_names should already be synced in length with self.num_decision_vars
        headers = list(self.custom_var_names) + ["Type", "RHS"]
        self.constraints_table.setColumnCount(self.num_decision_vars + 2)
        self.constraints_table.setHorizontalHeaderLabels(headers)
        
        for i in range(self.num_decision_vars + 2):
            if i < self.num_decision_vars:
                 self.constraints_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
            else:
                 self.constraints_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)

    def _populate_row_widgets(self, row_idx):
        for j in range(self.num_decision_vars):
            coeff_input = QLineEdit("0")
            coeff_input.setAlignment(Qt.AlignCenter)
            self.constraints_table.setCellWidget(row_idx, j, coeff_input)

        sense_combo = QComboBox()
        sense_combo.addItems(["<=", ">=", "=="])
        self.constraints_table.setCellWidget(row_idx, self.num_decision_vars, sense_combo)

        rhs_input = QLineEdit("0")
        rhs_input.setAlignment(Qt.AlignCenter)
        self.constraints_table.setCellWidget(row_idx, self.num_decision_vars + 1, rhs_input)

    def _update_num_vars_and_constraint_table(self, force_default=False):
        prev_num_vars = self.num_decision_vars
        new_num_vars_candidate = prev_num_vars # Start with current count
        
        if force_default:
            new_num_vars_candidate = 2 # Or your preferred default
            self.obj_coeffs_input.setText(",".join(["0"] * new_num_vars_candidate)) # Set a default for obj coeffs
            self.custom_var_names = [f"x{i+1}" for i in range(new_num_vars_candidate)] # Reset with defaults
        else:
            obj_coeffs_str = self.obj_coeffs_input.text().strip()
            if not obj_coeffs_str: # If empty, revert to a sensible default or current
                new_num_vars_candidate = self.num_decision_vars # Keep current, or set to a default like 1 or 2
                # QMessageBox.information(self, "Info", "Objective coefficients empty. Using previous/default variable count.")
            else:
                try:
                    coeffs = [float(c.strip()) for c in obj_coeffs_str.split(',') if c.strip()]
                    if not coeffs and obj_coeffs_str: # Input like "," or "a,b"
                         QMessageBox.warning(self, "Input Error", "Invalid objective coefficients. Please use comma-separated numbers.")
                         self.obj_coeffs_input.setText(",".join(["0"] * prev_num_vars)) # Revert to old valid or default
                         return # Do not change table structure on bad input
                    new_num_vars_candidate = len(coeffs) if coeffs else self.num_decision_vars # Use old if parse results in empty list but string wasn't empty
                except ValueError:
                    QMessageBox.warning(self, "Input Error", "Objective coefficients must be comma-separated numbers (e.g., 3,2,5).")
                    self.obj_coeffs_input.setText(",".join(["0"] * prev_num_vars))
                    return # Don't update if objective parsing fails

        if new_num_vars_candidate <= 0 : new_num_vars_candidate = 1 # Ensure at least one variable column

        # Only proceed with full table update if num_vars changed or it's a forced default
        if new_num_vars_candidate != self.num_decision_vars or force_default:
            self.num_decision_vars = new_num_vars_candidate
            
            # Adjust custom_var_names list to match new_num_vars, preserving existing names
            if force_default:
                 self.custom_var_names = [f"x{i+1}" for i in range(self.num_decision_vars)]
            else:
                current_custom_names_len = len(self.custom_var_names)
                if current_custom_names_len < self.num_decision_vars:
                    # Append new default names for added variables
                    for i in range(current_custom_names_len, self.num_decision_vars):
                        self.custom_var_names.append(f"x{i+1}")
                elif current_custom_names_len > self.num_decision_vars:
                    # Truncate if variables were removed
                    self.custom_var_names = self.custom_var_names[:self.num_decision_vars]
                # If current_custom_names_len == self.num_decision_vars, names are preserved.

            current_row_count = self.constraints_table.rowCount()
            # If table is empty and it's a default call (initial or clear), set to 3 rows
            if current_row_count == 0 and force_default: 
                current_row_count = 3 
            # If table is not empty, or it is empty but not a force_default (e.g. user deleted all rows then changed var count)
            # ensure at least one row if it becomes 0 otherwise.
            elif current_row_count == 0 and not force_default:
                current_row_count = 1 # Ensure at least one row for new variable structure
            
            self._setup_constraints_table_columns() 
            
            self.constraints_table.setRowCount(0) 
            self.constraints_table.setRowCount(current_row_count) 
            for i in range(current_row_count):
                self._populate_row_widgets(i)
        else:
            # If num_vars hasn't changed, still ensure headers are correct (e.g. after a rename)
            self._setup_constraints_table_columns()

    def add_constraint_row(self):
        row_count = self.constraints_table.rowCount()
        self.constraints_table.insertRow(row_count)
        self._populate_row_widgets(row_count)

    def remove_constraint_row(self):
        current_row = self.constraints_table.currentRow()
        if self.constraints_table.rowCount() == 0:
            return
        if current_row >= 0:
            self.constraints_table.removeRow(current_row)
        else: # If no row selected, remove the last one
            self.constraints_table.removeRow(self.constraints_table.rowCount() - 1)

    def _create_output_widgets(self):
        output_label = QLabel("Solution:")
        self.layout.addWidget(output_label)
        self.results_output = QTextEdit()
        self.results_output.setReadOnly(True)
        self.results_output.setFixedHeight(150) # Give a fixed height for the results
        self.layout.addWidget(self.results_output)

    def _create_control_buttons(self):
        button_layout = QHBoxLayout()
        self.solve_button = QPushButton("Solve")
        self.solve_button.clicked.connect(self.solve_problem)
        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.clear_inputs)
        button_layout.addWidget(self.solve_button)
        button_layout.addWidget(self.clear_button)
        self.layout.addLayout(button_layout)

    def clear_inputs(self):
        self.obj_coeffs_input.clear()
        self.results_output.clear()
        self.obj_sense_combo.setCurrentIndex(0)
        
        # Reset to default variable count and table structure
        self._update_num_vars_and_constraint_table(force_default=True)

    def solve_problem(self):
        try:
            # 1. Collect objective function info
            obj_sense = self.obj_sense_combo.currentText()
            obj_coeffs_str = self.obj_coeffs_input.text().strip()
            if not obj_coeffs_str:
                QMessageBox.warning(self, "Input Error", "Objective function coefficients cannot be empty.")
                return
            
            obj_coeffs = [float(c.strip()) for c in obj_coeffs_str.split(',') if c.strip()]
            if not obj_coeffs: # Handles cases like a single comma or non-numeric input that split to empty
                 QMessageBox.warning(self, "Input Error", "Invalid objective coefficients. Please provide numbers.")
                 return

            # Ensure self.num_decision_vars is up-to-date with what's in obj_coeffs_input
            # This might be slightly redundant if editingFinished always fires and updates correctly,
            # but good as a safeguard before solving.
            if len(obj_coeffs) != self.num_decision_vars:
                 # This case means user changed obj_coeffs but didn't tab out, then hit solve.
                 # We should force an update.
                 self._update_num_vars_and_constraint_table()
                 # Check again, if it's still mismatched, there might be an issue.
                 if len(obj_coeffs) != self.num_decision_vars:
                      QMessageBox.warning(self, "Input Sync Error", "Mismatch in variable count. Please re-check objective coefficients or press Tab/Enter in that field.")
                      return

            # 2. Collect constraints
            constraint_matrix = []
            constraint_senses = []
            constraint_rhs = []
            active_constraints_found = 0

            for i in range(self.constraints_table.rowCount()):
                current_row_coeffs_str = []
                is_row_effectively_empty = True # Assume empty until proven otherwise

                for j in range(self.num_decision_vars):
                    cell_w = self.constraints_table.cellWidget(i, j)
                    val_str = cell_w.text().strip() if cell_w and isinstance(cell_w, QLineEdit) else "0"
                    current_row_coeffs_str.append(val_str)
                    if val_str and float(val_str) != 0: # Consider it non-empty if a coeff is non-zero
                        is_row_effectively_empty = False
                
                rhs_widget = self.constraints_table.cellWidget(i, self.num_decision_vars + 1)
                rhs_val_str = rhs_widget.text().strip() if rhs_widget and isinstance(rhs_widget, QLineEdit) else "0"
                if rhs_val_str and float(rhs_val_str) != 0 : # Also consider non-empty if RHS is non-zero
                     is_row_effectively_empty = False
                
                # If all coefficient inputs are "0" or empty, AND RHS is "0" or empty, skip this row.
                # This check needs to be careful. A row like "0,0 <= 0" is a valid (though perhaps trivial) constraint.
                # Let's refine: skip if all original text entries for coeffs AND RHS were empty.
                # For now, the logic is: if after defaulting empty to "0", all coeffs are 0 and RHS is 0, it might be an inactive row.
                # A better check: if all QLineEdit.text() were originally empty.
                # The current 'is_row_effectively_empty' might skip "0x1 + 0x2 <= 0" which is valid.
                # Let's collect all rows and let solver handle trivial constraints, unless all inputs for that row were literally empty.

                # Collect actual values, defaulting empty to 0
                actual_coeffs_for_row = []
                for val_str in current_row_coeffs_str:
                    try:
                        actual_coeffs_for_row.append(float(val_str) if val_str else 0.0)
                    except ValueError:
                        QMessageBox.warning(self, "Input Error", f"Constraint {i+1}, coefficient '{val_str}' is not a valid number.")
                        return

                # Check if all original text inputs for this row were empty
                all_original_text_empty = all(not self.constraints_table.cellWidget(i,j).text().strip() for j in range(self.num_decision_vars)) and \
                                          not self.constraints_table.cellWidget(i, self.num_decision_vars + 1).text().strip()
                if all_original_text_empty:
                    continue # Skip row if all its original text inputs were empty

                active_constraints_found +=1
                constraint_matrix.append(actual_coeffs_for_row)

                sense_widget = self.constraints_table.cellWidget(i, self.num_decision_vars)
                if sense_widget is None: continue # Should not happen
                constraint_senses.append(sense_widget.currentText())

                try:
                    constraint_rhs.append(float(rhs_val_str) if rhs_val_str else 0.0)
                except ValueError:
                    QMessageBox.warning(self, "Input Error", f"Constraint {i+1}, RHS '{rhs_val_str}' is not a valid number.")
                    return

            if active_constraints_found == 0:
                QMessageBox.warning(self, "Input Error", "Please enter at least one valid constraint.")
                return

            problem_input = LPProblemInput(
                objective_coefficients=obj_coeffs,
                objective_sense=obj_sense,
                constraint_matrix=constraint_matrix,
                constraint_senses=constraint_senses,
                constraint_rhs=constraint_rhs
            )

            solution = solve_lp_problem(problem_input)

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

        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Invalid numeric input. Please check all numbers: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Solver Error", f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc() # For debugging

# 主函数，用于测试运行 MainWindow
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_()) 