"""
APLICACIÓN INTERACTIVA DE REDES BAYESIANAS
Interfaz gráfica PyQt6 | Inferencia: Variable Elimination + Min-Fill
Almacenamiento: JSON
"""

import sys
import json
import math
import copy
from itertools import product

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem,
    QGraphicsItem, QGraphicsPathItem, QFrame,
    QLabel, QLineEdit, QMessageBox, QInputDialog, QDialog,
    QDialogButtonBox, QTableWidget, QTableWidgetItem, QComboBox,
    QFormLayout, QHeaderView, QFileDialog, QScrollArea
)
from PyQt6.QtCore import Qt, QPointF, QLineF, QRectF, QTimer
from PyQt6.QtGui import (
    QPen, QBrush, QColor, QFont, QPainter, QPolygonF, QImage,
    QPageLayout, QShortcut, QKeySequence, QPainterPath
)
from PyQt6.QtPrintSupport import QPrinter, QPrintDialog


# ==================================================
# 1. FACTORES
# ==================================================

class Factor:
    def __init__(self, vars_list, cardinality, table):
        self.vars = list(vars_list)
        self.cardinality = cardinality
        self.table = dict(table)

    def restrict(self, var, value):
        if var not in self.vars:
            return self
        idx = self.vars.index(var)
        new_vars = [v for v in self.vars if v != var]
        new_table = {}
        for key, val in self.table.items():
            if key[idx] == value:
                new_key = key[:idx] + key[idx + 1:]
                new_table[new_key] = val
        return Factor(new_vars, self.cardinality, new_table)

    def multiply(self, other):
        new_vars = list(self.vars)
        for var in other.vars:
            if var not in new_vars:
                new_vars.append(var)

        new_table = {}

        def assignments(index=0, current=None):
            if current is None:
                current = []
            if index == len(new_vars):
                yield tuple(current)
                return
            var = new_vars[index]
            for state in self.cardinality[var]:
                current.append(state)
                yield from assignments(index + 1, current)
                current.pop()

        for values in assignments():
            assignment = dict(zip(new_vars, values))
            key1 = tuple(assignment[v] for v in self.vars)
            key2 = tuple(assignment[v] for v in other.vars)
            new_table[values] = (
                self.table.get(key1, 0.0) *
                other.table.get(key2, 0.0)
            )
        return Factor(new_vars, self.cardinality, new_table)

    def sum_out(self, var):
        if var not in self.vars:
            return self
        idx = self.vars.index(var)
        new_vars = [v for v in self.vars if v != var]
        new_table = {}
        for key, value in self.table.items():
            new_key = key[:idx] + key[idx + 1:]
            new_table[new_key] = new_table.get(new_key, 0.0) + value
        return Factor(new_vars, self.cardinality, new_table)

    def normalize(self):
        total = sum(self.table.values())
        if total <= 0 or not math.isfinite(total):
            raise ValueError(
                "La probabilidad total es cero o inválida; no se puede normalizar."
            )
        for key in self.table:
            self.table[key] /= total
        return self


# ==================================================
# 2. ORDEN DE ELIMINACIÓN MIN-FILL
# ==================================================

def min_fill_order(factors, hidden_vars, cardinality):
    """
    Calcula un orden Min-Fill.
    Criterio principal: menor número de aristas nuevas entre vecinos.
    Desempates: menor tamaño estimado del factor, menor grado y orden original.
    """
    graph = {var: set() for var in cardinality}

    # Cada factor induce un subgrafo completo entre sus variables.
    for factor in factors:
        variables = factor.vars
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                a, b = variables[i], variables[j]
                graph[a].add(b)
                graph[b].add(a)

    remaining = list(hidden_vars)
    order = []

    while remaining:
        best_var = None
        best_score = None

        for var in remaining:
            neighbors = list(graph[var])
            fill_count = 0

            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    a, b = neighbors[i], neighbors[j]
                    if b not in graph[a]:
                        fill_count += 1

            estimated_size = len(cardinality[var])
            for neighbor in neighbors:
                estimated_size *= len(cardinality[neighbor])

            score = (
                fill_count,
                estimated_size,
                len(neighbors),
                hidden_vars.index(var)
            )

            if best_score is None or score < best_score:
                best_score = score
                best_var = var

        order.append(best_var)

        # Al eliminar best_var, sus vecinos quedan conectados entre sí.
        neighbors = list(graph[best_var])
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a, b = neighbors[i], neighbors[j]
                graph[a].add(b)
                graph[b].add(a)

        for neighbor in neighbors:
            graph[neighbor].discard(best_var)
        graph[best_var].clear()
        remaining.remove(best_var)

    return order


# ==================================================
# 3. VARIABLE ELIMINATION
# ==================================================

def execute_elimination(factors, elimination_order, query_vars, cardinality):
    """Ejecuta un orden y registra el tamaño máximo de los factores."""
    # Copiar los factores para que las dos ejecuciones sean independientes.
    working_factors = [
        Factor(factor.vars, cardinality, factor.table.copy())
        for factor in factors
    ]

    max_entries = max((len(f.table) for f in working_factors), default=0)
    max_variables = max((len(f.vars) for f in working_factors), default=0)

    for var in elimination_order:
        relevant = [f for f in working_factors if var in f.vars]
        others = [f for f in working_factors if var not in f.vars]

        if not relevant:
            working_factors = others
            continue

        product_factor = relevant[0]
        for factor in relevant[1:]:
            product_factor = product_factor.multiply(factor)

        max_entries = max(max_entries, len(product_factor.table))
        max_variables = max(max_variables, len(product_factor.vars))

        summed_factor = product_factor.sum_out(var)
        max_entries = max(max_entries, len(summed_factor.table))
        max_variables = max(max_variables, len(summed_factor.vars))
        working_factors = others + [summed_factor]

    if not working_factors:
        raise ValueError("No existen factores para resolver la consulta.")

    final_factor = working_factors[0]
    for factor in working_factors[1:]:
        final_factor = final_factor.multiply(factor)
        max_entries = max(max_entries, len(final_factor.table))
        max_variables = max(max_variables, len(final_factor.vars))

    for var in list(final_factor.vars):
        if var not in query_vars:
            final_factor = final_factor.sum_out(var)

    final_factor.normalize()
    stats = {
        "max_entries": max_entries,
        "max_variables": max_variables
    }
    return final_factor, stats


def variable_elimination(net_data, query_vars, evidence):
    nodes = net_data["nodes"]
    cardinality = {node["name"]: node["states"] for node in nodes}

    if not query_vars:
        raise ValueError("Debes seleccionar una variable de consulta.")
    if len(query_vars) != len(set(query_vars)):
        raise ValueError("No puedes repetir variables de consulta.")
    if any(var not in cardinality for var in query_vars):
        raise ValueError("La variable consultada no existe.")
    if any(var not in cardinality for var in evidence):
        raise ValueError("Una variable de evidencia no existe.")
    if set(query_vars) & set(evidence):
        raise ValueError(
            "La variable consultada no puede ser también evidencia."
        )

    for var, value in evidence.items():
        if value not in cardinality[var]:
            raise ValueError(f"El estado {value} no es válido para {var}.")

    factors = []
    for node in nodes:
        var = node["name"]
        parents = node.get("parents", [])
        factor_vars = parents + [var]
        table = {}

        for entry in node["cpt"]:
            parent_assignment = entry.get("parents", {})
            for state, probability in entry["probabilities"].items():
                key = tuple(
                    parent_assignment[parent] for parent in parents
                ) + (state,)
                table[key] = float(probability)

        factors.append(Factor(factor_vars, cardinality, table))

    for var, value in evidence.items():
        factors = [factor.restrict(var, value) for factor in factors]

    all_vars = list(cardinality.keys())
    hidden_vars = [
        var for var in all_vars
        if var not in query_vars and var not in evidence
    ]

    # Se comparan dos órdenes: el orden de creación de los nodos y Min-Fill.
    original_order = hidden_vars.copy()
    min_fill = min_fill_order(factors, hidden_vars, cardinality)

    original_result, original_stats = execute_elimination(
        factors, original_order, query_vars, cardinality
    )
    min_fill_result, min_fill_stats = execute_elimination(
        factors, min_fill, query_vars, cardinality
    )

    # Ambos órdenes deben producir la misma distribución de probabilidad.
    if original_result.vars != min_fill_result.vars:
        raise ValueError("Los órdenes de eliminación produjeron resultados incompatibles.")
    for key, probability in original_result.table.items():
        if not math.isclose(
            probability, min_fill_result.table.get(key, 0.0),
            rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("La comparación de órdenes produjo probabilidades diferentes.")

    comparison = {
        "original": original_stats,
        "min_fill": min_fill_stats
    }
    return min_fill_result, min_fill, original_order, comparison


# ==================================================
# 4. NODO GRÁFICO
# ==================================================

class NodeItem(QGraphicsEllipseItem):
    """Nodo gráfico movible; su etiqueta se mueve junto con el círculo."""

    def __init__(self, x, y, name, window):
        super().__init__(0, 0, 100, 100)

        self.name = name
        self.window = window
        self._move_update_pending = False

        self.setPen(QPen(QColor("#2563EB"), 2))
        self.setBrush(QBrush(QColor("#DBEAFE")))
        self.setData(0, "node")
        self.setPos(x, y)

        from PyQt6.QtWidgets import QGraphicsTextItem

        self.label = QGraphicsTextItem(name, self)
        self.label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.label.setDefaultTextColor(QColor("#1E3A8A"))

        bounds = self.label.boundingRect()
        self.label.setPos(
            (100 - bounds.width()) / 2,
            (100 - bounds.height()) / 2
        )
        self.label.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True
        )
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True
        )
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True
        )

        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def itemChange(self, change, value):
        result = super().itemChange(change, value)

        if (
            change ==
            QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
        ):
            # Se difiere la actualización para no modificar la escena
            # dentro del propio evento de cambio de posición.
            if not self._move_update_pending:
                self._move_update_pending = True
                QTimer.singleShot(0, self._finish_position_update)

        return result

    def _finish_position_update(self):
        self._move_update_pending = False

        if self.scene() is not None:
            self.window.node_moved(self.name, self.pos())

    def mousePressEvent(self, event):
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, event):
        self.window.edit_node(self.name)
        event.accept()


class ConnectionItem(QGraphicsPathItem):
    """Flecha seleccionable que representa una conexión padre → hijo."""

    def __init__(self, parent_name, child_name, start_point, end_point, ux, uy):
        path = QPainterPath()
        path.moveTo(start_point)
        path.lineTo(end_point)

        arrow_size = 12
        base = QPointF(end_point.x() - ux * arrow_size,
                       end_point.y() - uy * arrow_size)
        p1 = QPointF(base.x() - uy * 6, base.y() + ux * 6)
        p2 = QPointF(base.x() + uy * 6, base.y() - ux * 6)
        path.moveTo(end_point)
        path.lineTo(p1)
        path.lineTo(p2)
        path.closeSubpath()

        super().__init__(path)
        self.parent_name = parent_name
        self.child_name = child_name
        self.setData(0, "connection")
        self.setData(1, (parent_name, child_name))
        self.setPen(QPen(QColor("#0F9F78"), 2.5))
        self.setBrush(QBrush(QColor("#0F9F78")))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setZValue(-1)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            if value:
                self.setPen(QPen(QColor("#DC2626"), 4))
                self.setBrush(QBrush(QColor("#DC2626")))
            else:
                self.setPen(QPen(QColor("#0F9F78"), 2.5))
                self.setBrush(QBrush(QColor("#0F9F78")))
        return result


# Estilo compartido para las ventanas secundarias, coherente con la ventana principal.
DIALOG_STYLE = """
    QDialog { background-color: #F1F5F9; color: #1E293B; }
    QLabel { color: #334155; font-size: 12px; }
    QLabel#dialogTitle { color: #1E3A8A; font-size: 17px; font-weight: 800; padding: 4px 0 8px 0; }
    QLabel#dialogSection { color: #64748B; font-size: 10px; font-weight: 700; letter-spacing: 1px; padding-top: 5px; }
    QLineEdit, QComboBox { background: #FFFFFF; color: #1E293B; border: 1px solid #CBD5E1; border-radius: 7px; padding: 7px 9px; min-height: 20px; selection-background-color: #2563EB; }
    QLineEdit:focus, QComboBox:focus { border: 1px solid #2563EB; }
    QComboBox QAbstractItemView { background: #FFFFFF; color: #1E293B; selection-background-color: #DBEAFE; selection-color: #1E3A8A; border: 1px solid #CBD5E1; }
    QTableWidget { background: #FFFFFF; alternate-background-color: #F8FAFC; color: #1E293B; gridline-color: #E2E8F0; border: 1px solid #CBD5E1; border-radius: 9px; selection-background-color: #DBEAFE; selection-color: #1E3A8A; }
    QHeaderView::section { background: #E2E8F0; color: #334155; border: none; border-right: 1px solid #CBD5E1; border-bottom: 1px solid #CBD5E1; padding: 8px; font-weight: 700; }
    QPushButton { background: #E2E8F0; color: #1E293B; border: none; border-radius: 8px; padding: 8px 14px; font-size: 12px; font-weight: 600; min-width: 76px; }
    QPushButton:hover { background: #CBD5E1; }
    QPushButton:default { background: #2563EB; color: #FFFFFF; }
    QPushButton:default:hover { background: #1D4ED8; }
    QDialogButtonBox QPushButton[text="Calcular"], QPushButton#primaryDialogButton { background: #2563EB; color: #FFFFFF; }
    QDialogButtonBox QPushButton[text="Calcular"]:hover, QPushButton#primaryDialogButton:hover { background: #1D4ED8; }
    QScrollBar:vertical { background: #E2E8F0; width: 10px; margin: 2px; border-radius: 5px; }
    QScrollBar::handle:vertical { background: #94A3B8; min-height: 24px; border-radius: 5px; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""

def apply_dialog_style(dialog):
    dialog.setStyleSheet(DIALOG_STYLE)

# ==================================================
# 5. EDITOR CPT
# ==================================================

class CPTDialog(QDialog):
    def __init__(self, name, parents, parent_states, existing_data=None):
        super().__init__()
        apply_dialog_style(self)
        self.setWindowTitle(f"Configurar nodo: {name}")
        self.resize(750, 450)
        self.parents = parents
        self.parent_states = parent_states
        self.combinations = []
        self.result = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        dialog_title = QLabel(f"Configurar nodo: {name}")
        dialog_title.setObjectName("dialogTitle")
        layout.addWidget(dialog_title)
        states_label = QLabel("ESTADOS DE LA VARIABLE")
        states_label.setObjectName("dialogSection")
        layout.addWidget(states_label)
        layout.addWidget(QLabel("Estados separados por comas:"))

        self.states_input = QLineEdit()
        self.states_input.setText(
            ", ".join(existing_data["states"])
            if existing_data else "Sí, No"
        )
        layout.addWidget(self.states_input)

        self.generate_button = QPushButton(
            "Generar tabla de probabilidades"
        )
        layout.addWidget(self.generate_button)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(True)
        layout.addWidget(self.table)
        self.generate_button.clicked.connect(self.generate_table)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.generate_table()
        if existing_data:
            self.load_existing_data(existing_data)

    def generate_table(self):
        states = [
            state.strip()
            for state in self.states_input.text().split(",")
            if state.strip()
        ]

        if len(states) < 2 or len(states) != len(set(states)):
            QMessageBox.warning(
                self, "Estados inválidos",
                "Introduce al menos dos estados diferentes."
            )
            return

        for parent in self.parents:
            if parent not in self.parent_states:
                QMessageBox.warning(
                    self, "Padre sin configurar",
                    f"Configura primero el nodo {parent}."
                )
                return

        self.states = states
        combinations = list(product(*[
            self.parent_states[parent] for parent in self.parents
        ]))
        if not self.parents:
            combinations = [()]
        self.combinations = combinations

        columns = len(self.parents) + len(states)
        self.table.clear()
        self.table.setRowCount(len(combinations))
        self.table.setColumnCount(columns)
        self.table.setHorizontalHeaderLabels(self.parents + states)

        for row, combination in enumerate(combinations):
            for col, value in enumerate(combination):
                item = QTableWidgetItem(value)
                item.setFlags(
                    item.flags() & ~Qt.ItemFlag.ItemIsEditable
                )
                self.table.setItem(row, col, item)

            initial = 1.0 / len(states)
            for index in range(len(states)):
                col = len(self.parents) + index
                self.table.setItem(
                    row, col, QTableWidgetItem(str(round(initial, 4)))
                )

        self.table.resizeColumnsToContents()

    def load_existing_data(self, data):
        for row, combination in enumerate(self.combinations):
            for entry in data["cpt"]:
                assignment = entry.get("parents", {})
                if all(
                    assignment.get(parent) == value
                    for parent, value in zip(self.parents, combination)
                ):
                    for index, state in enumerate(self.states):
                        col = len(self.parents) + index
                        probability = entry["probabilities"].get(state, 0)
                        self.table.setItem(
                            row, col, QTableWidgetItem(str(probability))
                        )
                    break

    def validate_and_accept(self):
        cpt = []

        for row, combination in enumerate(self.combinations):
            probabilities = {}
            total = 0.0

            for index, state in enumerate(self.states):
                col = len(self.parents) + index
                item = self.table.item(row, col)

                try:
                    probability = float(item.text())
                except (ValueError, AttributeError):
                    QMessageBox.warning(
                        self, "Error",
                        "Todas las probabilidades deben ser números."
                    )
                    return

                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    QMessageBox.warning(
                        self, "Error",
                        "Las probabilidades deben estar entre 0 y 1."
                    )
                    return

                probabilities[state] = probability
                total += probability

            if not math.isclose(total, 1.0, abs_tol=0.0001):
                QMessageBox.warning(
                    self, "Error",
                    f"La suma de probabilidades de la fila {row + 1} "
                    f"es {total:.4f}; debe ser 1."
                )
                return

            entry = {"probabilities": probabilities}
            if self.parents:
                entry["parents"] = dict(zip(self.parents, combination))
            cpt.append(entry)

        self.result = {"states": self.states, "cpt": cpt}
        self.accept()


# ==================================================
# 6. DIÁLOGO DE CONSULTAS
# ==================================================

class QueryDialog(QDialog):
    def __init__(self, node_data):
        super().__init__()
        apply_dialog_style(self)
        self.setWindowTitle("Realizar consulta bayesiana")
        self.resize(400, 450)
        self.node_data = node_data
        self.evidence_inputs = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        dialog_title = QLabel("Consulta de probabilidad")
        dialog_title.setObjectName("dialogTitle")
        layout.addWidget(dialog_title)
        query_label = QLabel("VARIABLE A CONSULTAR")
        query_label.setObjectName("dialogSection")
        layout.addWidget(query_label)
        layout.addWidget(QLabel(
            "Selecciona la variable que deseas consultar:"
        ))
        self.query_combo = QComboBox()
        self.query_combo.addItems(list(node_data.keys()))
        layout.addWidget(self.query_combo)

        evidence_label = QLabel("EVIDENCIAS")
        evidence_label.setObjectName("dialogSection")
        layout.addWidget(evidence_label)
        layout.addWidget(QLabel(
            "Selecciona los valores conocidos (evidencias):"
        ))
        self.evidence_layout = QFormLayout()
        layout.addLayout(self.evidence_layout)

        self.query_combo.currentTextChanged.connect(self.refresh_evidence)
        self.refresh_evidence()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(
            QDialogButtonBox.StandardButton.Ok
        ).setText("Calcular")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def refresh_evidence(self):
        while self.evidence_layout.count():
            item = self.evidence_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.evidence_inputs.clear()
        query_variable = self.query_combo.currentText()

        for name, data in self.node_data.items():
            combo = QComboBox()
            if name == query_variable:
                combo.addItem("Variable consultada")
                combo.setEnabled(False)
            else:
                combo.addItem("Sin evidencia")
                combo.addItems(data["states"])

            self.evidence_layout.addRow(f"{name}:", combo)
            self.evidence_inputs[name] = combo

    def get_query(self):
        query_variable = self.query_combo.currentText()
        evidence = {}

        for name, combo in self.evidence_inputs.items():
            if name == query_variable:
                continue
            value = combo.currentText()
            if value != "Sin evidencia":
                evidence[name] = value

        return query_variable, evidence


# ==================================================
# 7. RESULTADOS
# ==================================================

class ResultsDialog(QDialog):
    def __init__(self, variable, factor, elimination_order,
                 original_order, comparison):
        super().__init__()
        apply_dialog_style(self)
        self.setWindowTitle("Resultado de la consulta")
        self.resize(500, 350)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        title = QLabel(f"Resultado de consulta: {variable}")
        title.setObjectName("dialogTitle")
        layout.addWidget(title)

        original_text = (
            " → ".join(original_order)
            if original_order else "No se eliminaron variables ocultas."
        )
        min_fill_text = (
            " → ".join(elimination_order)
            if elimination_order else "No se eliminaron variables ocultas."
        )
        order_label = QLabel(
            f"Orden original (creación de nodos): {original_text}\n"
            f"Orden Min-Fill: {min_fill_text}"
        )
        order_label.setWordWrap(True)
        layout.addWidget(order_label)

        original_stats = comparison["original"]
        min_fill_stats = comparison["min_fill"]
        comparison_label = QLabel(
            "Comparación del tamaño máximo de los factores intermedios\n"
            f"Orden original: {original_stats['max_entries']} filas, "
            f"hasta {original_stats['max_variables']} variables\n"
            f"Min-Fill: {min_fill_stats['max_entries']} filas, "
            f"hasta {min_fill_stats['max_variables']} variables"
        )
        comparison_label.setWordWrap(True)
        layout.addWidget(comparison_label)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels([
            "Estado", "Probabilidad", "Porcentaje"
        ])

        states = factor.cardinality[variable]
        self.table.setRowCount(len(states))

        # La consulta de esta interfaz es de una sola variable.
        if factor.vars != [variable]:
            raise ValueError(
                "El resultado no tiene la estructura esperada "
                "para una consulta de una variable."
            )

        for row, state in enumerate(states):
            probability = factor.table.get((state,), 0.0)
            self.table.setItem(row, 0, QTableWidgetItem(state))
            self.table.setItem(
                row, 1, QTableWidgetItem(f"{probability:.6f}")
            )
            self.table.setItem(
                row, 2, QTableWidgetItem(f"{probability * 100:.2f}%")
            )

        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.table)

        close_button = QPushButton("Cerrar")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)


# ==================================================
# 8. VENTANA PRINCIPAL
# ==================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Redes Bayesianas | Variable Elimination + Min-Fill")
        self.resize(1280, 760)
        self.setMinimumSize(1050, 650)

        self.scene = QGraphicsScene()
        self.scene.setSceneRect(0, 0, 1000, 700)
        self.graph_view = QGraphicsView(self.scene)
        self.graph_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graph_view.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.graph_view.setBackgroundBrush(QBrush(QColor("#F8FAFC")))
        self.graph_view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.graph_view.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate
        )

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(14)

        # Panel lateral con agrupación visual de herramientas.
        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel.setFixedWidth(250)
        side_panel = QVBoxLayout(panel)
        side_panel.setContentsMargins(16, 18, 16, 16)
        side_panel.setSpacing(9)

        title = QLabel("REDES BAYESIANAS")
        title.setObjectName("appTitle")
        subtitle = QLabel("Editor de redes probabilísticas")
        subtitle.setObjectName("appSubtitle")
        side_panel.addWidget(title)
        side_panel.addWidget(subtitle)
        side_panel.addSpacing(12)

        self.add_button = QPushButton("＋  Agregar nodo")
        self.connect_button = QPushButton("↗  Conectar nodos")
        self.delete_connection_button = QPushButton("⌫  Eliminar selección")
        self.query_button = QPushButton("▤  Realizar consulta")
        self.save_button = QPushButton("Guardar red")
        self.load_button = QPushButton("Cargar red")
        self.clear_button = QPushButton("Limpiar red")
        self.export_button = QPushButton("Exportar red a PNG")
        self.print_button = QPushButton("Imprimir red")

        for button in (self.add_button, self.connect_button, self.delete_connection_button, self.query_button,
                       self.save_button, self.load_button, self.clear_button,
                       self.export_button, self.print_button):
            button.setMinimumHeight(38)
            button.setCursor(Qt.CursorShape.PointingHandCursor)

        self.add_button.setObjectName("primaryButton")
        self.query_button.setObjectName("primaryButton")
        self.clear_button.setObjectName("dangerButton")

        def section_label(text):
            label = QLabel(text.upper())
            label.setObjectName("sectionLabel")
            side_panel.addSpacing(8)
            side_panel.addWidget(label)

        section_label("Construcción")
        for button in (self.add_button, self.connect_button, self.delete_connection_button):
            side_panel.addWidget(button)
        section_label("Inferencia")
        side_panel.addWidget(self.query_button)
        section_label("Archivo y salida")
        for button in (self.save_button, self.load_button,
                       self.export_button, self.print_button):
            side_panel.addWidget(button)
        side_panel.addSpacing(8)
        side_panel.addWidget(self.clear_button)
        side_panel.addStretch()

        # Consejo desplazable: permite leer todo el texto aunque el panel sea pequeño.
        hint = QLabel(
            "Consejo: arrastra los nodos para organizar la red.\n\n"
            "Doble clic en un nodo para editar su CPT.\n\n"
            "Selecciona uno o varios nodos y/o flechas; pulsa Backspace para eliminarlos.\n\n"
            "Ctrl+Z deshace las eliminaciones.\n\n"
            "Puedes seleccionar varios elementos manteniendo presionada la tecla Ctrl."
        )
        hint.setObjectName("helpHint")
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        hint_scroll = QScrollArea()
        hint_scroll.setObjectName("helpScroll")
        hint_scroll.setWidgetResizable(True)
        hint_scroll.setFrameShape(QFrame.Shape.NoFrame)
        hint_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        hint_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        hint_scroll.setMinimumHeight(105)
        hint_scroll.setMaximumHeight(155)
        hint_scroll.setWidget(hint)
        side_panel.addWidget(hint_scroll)

        main_layout.addWidget(panel)
        main_layout.addWidget(self.graph_view, 1)

        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #F1F5F9; color: #1E293B; }
            QFrame#sidePanel { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; }
            QLabel#appTitle { color: #1E3A8A; font-size: 17px; font-weight: 800; }
            QLabel#appSubtitle { color: #64748B; font-size: 11px; }
            QLabel#sectionLabel { color: #64748B; font-size: 10px; font-weight: 700; letter-spacing: 1px; }
            QScrollArea#helpScroll { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; }
            QLabel#helpHint { color: #64748B; background: #F8FAFC; padding: 10px; font-size: 11px; }
            QScrollArea#helpScroll QScrollBar:vertical { background: #E2E8F0; width: 9px; margin: 3px; border-radius: 4px; }
            QScrollArea#helpScroll QScrollBar::handle:vertical { background: #94A3B8; min-height: 22px; border-radius: 4px; }
            QScrollArea#helpScroll QScrollBar::handle:vertical:hover { background: #64748B; }
            QScrollArea#helpScroll QScrollBar::add-line:vertical, QScrollArea#helpScroll QScrollBar::sub-line:vertical { height: 0px; }
            QPushButton { background: #E2E8F0; color: #1E293B; border: none; border-radius: 8px; padding: 8px 10px; text-align: left; font-size: 12px; font-weight: 600; }
            QPushButton:hover { background: #CBD5E1; }
            QPushButton#primaryButton { background: #2563EB; color: white; }
            QPushButton#primaryButton:hover { background: #1D4ED8; }
            QPushButton#dangerButton { background: #FEE2E2; color: #B91C1C; }
            QPushButton#dangerButton:hover { background: #FECACA; }
            QGraphicsView { background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 14px; }
            QStatusBar { background: #FFFFFF; color: #475569; border-top: 1px solid #E2E8F0; }
            QDialog { background: #FFFFFF; }
        """)

        # Backspace elimina toda la selección; Ctrl+Z revierte eliminaciones.
        self.delete_selection_shortcut = QShortcut(QKeySequence("Backspace"), self.graph_view)
        self.delete_selection_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.delete_selection_shortcut.activated.connect(self.delete_selected_items)
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self.graph_view)
        self.undo_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.undo_shortcut.activated.connect(self.undo_last_deletion)
        self.graph_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.add_button.clicked.connect(self.add_node)
        self.connect_button.clicked.connect(self.connect_nodes)
        self.delete_connection_button.clicked.connect(self.delete_selected_items)
        self.query_button.clicked.connect(self.run_query)
        self.save_button.clicked.connect(self.save_network)
        self.load_button.clicked.connect(self.load_network)
        self.clear_button.clicked.connect(self.clear_network)
        self.export_button.clicked.connect(self.export_network_png)
        self.print_button.clicked.connect(self.print_network)

        self.statusBar().showMessage(
            "Listo | Arrastra los nodos para organizar la red"
        )
        self.node_count = 0
        self.node_names = []
        self.connections = []
        self.node_positions = {}
        self.node_data = {}
        self.deletion_history = []
        self.max_undo_steps = 100

    # ---------------- EXPORTAR E IMPRIMIR RED ----------------

    def network_source_rect(self):
        """Obtiene los límites reales de los elementos dibujados, con margen."""
        rect = self.scene.itemsBoundingRect()
        if rect.isNull() or rect.isEmpty():
            return rect
        return rect.adjusted(-35, -35, 35, 35)

    def export_network_png(self):
        if not self.node_names:
            QMessageBox.warning(
                self, "Red vacía", "Primero debes crear al menos un nodo."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exportar red bayesiana", "red_bayesiana.png",
            "Imagen PNG (*.png)"
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".png"):
            file_path += ".png"

        source = self.network_source_rect()
        if source.isNull() or source.isEmpty():
            QMessageBox.warning(
                self, "Sin contenido", "No hay elementos para exportar."
            )
            return

        # Escala 2x para obtener una imagen con buena resolución.
        scale = 2
        image = QImage(
            max(1, int(source.width() * scale)),
            max(1, int(source.height() * scale)),
            QImage.Format.Format_ARGB32
        )
        image.fill(QColor("white"))

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.scene.render(
            painter, QRectF(image.rect()), source, Qt.AspectRatioMode.KeepAspectRatio
        )
        painter.end()

        if not image.save(file_path, "PNG"):
            QMessageBox.critical(
                self, "Error al exportar", "No se pudo guardar la imagen PNG."
            )
            return

        QMessageBox.information(
            self, "Exportación completada",
            f"La red se exportó correctamente en:\n{file_path}"
        )
        self.statusBar().showMessage("Red exportada a PNG")

    def print_network(self):
        if not self.node_names:
            QMessageBox.warning(
                self, "Red vacía", "Primero debes crear al menos un nodo."
            )
            return

        source = self.network_source_rect()
        if source.isNull() or source.isEmpty():
            QMessageBox.warning(
                self, "Sin contenido", "No hay elementos para imprimir."
            )
            return

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setPageOrientation(QPageLayout.Orientation.Landscape)
        dialog = QPrintDialog(printer, self)
        dialog.setWindowTitle("Imprimir red bayesiana")
        if dialog.exec() != QPrintDialog.DialogCode.Accepted:
            return

        # Algunos controladores modifican la orientación al cerrar el diálogo.
        # Se vuelve a fijar horizontal para garantizar el formato apaisado.
        printer.setPageOrientation(QPageLayout.Orientation.Landscape)

        painter = QPainter()
        if not painter.begin(printer):
            QMessageBox.critical(
                self, "Error de impresión",
                "No se pudo iniciar el trabajo de impresión."
            )
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Área imprimible real, en píxeles de la impresora.
        page_rect = QRectF(printer.pageRect(QPrinter.Unit.DevicePixel))

        # Calcula un rectángulo centrado que contiene toda la red sin deformarla.
        scale = min(
            page_rect.width() / source.width(),
            page_rect.height() / source.height()
        )
        target_width = source.width() * scale
        target_height = source.height() * scale
        target_rect = QRectF(
            page_rect.left() + (page_rect.width() - target_width) / 2,
            page_rect.top() + (page_rect.height() - target_height) / 2,
            target_width,
            target_height
        )

        self.scene.render(
            painter, target_rect, source, Qt.AspectRatioMode.KeepAspectRatio
        )
        painter.end()

        self.statusBar().showMessage("Trabajo de impresión enviado")

    # ---------------- AGREGAR NODO ----------------

    def add_node(self):
        name, accepted = QInputDialog.getText(
            self, "Nuevo nodo", "Nombre de la variable:"
        )
        if not accepted:
            return

        name = name.strip()
        if not name:
            QMessageBox.warning(self, "Error", "El nombre no puede estar vacío.")
            return
        if name in self.node_names:
            QMessageBox.warning(self, "Error", "Ese nodo ya existe.")
            return
        if self.node_count >= 10:
            QMessageBox.warning(
                self, "Límite", "La red admite un máximo de 10 nodos."
            )
            return

        self.node_names.append(name)
        x = 100 + (self.node_count % 3) * 200
        y = 100 + (self.node_count // 3) * 150

        node_item = NodeItem(x, y, name, self)
        self.scene.addItem(node_item)
        self.node_positions[name] = QPointF(x + 50, y + 50)
        self.update_scene_bounds()

        self.node_count += 1
        self.statusBar().showMessage(f"Nodos creados: {self.node_count}/10")

    def node_moved(self, name, top_left):
        """Actualiza la posición lógica y las flechas al arrastrar un nodo."""
        if name not in self.node_positions:
            return
        self.node_positions[name] = QPointF(top_left.x() + 50, top_left.y() + 50)
        self.draw_connections()
        self.update_scene_bounds()

    def update_scene_bounds(self):
        """Amplía el área desplazable según la extensión actual de la red."""
        rect = self.scene.itemsBoundingRect().adjusted(-100, -100, 100, 100)
        if not rect.isNull() and not rect.isEmpty():
            self.scene.setSceneRect(rect)

    # ---------------- EDITAR NODO ----------------

    def edit_node(self, name):
        parents = [
            origin for origin, destination in self.connections
            if destination == name
        ]
        parent_states = {}

        for parent in parents:
            if parent not in self.node_data:
                QMessageBox.warning(
                    self, "Configuración pendiente",
                    f"Primero configura el nodo padre: {parent}."
                )
                return
            parent_states[parent] = self.node_data[parent]["states"]

        existing_data = self.node_data.get(name)
        dialog = CPTDialog(name, parents, parent_states, existing_data)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            old_states = existing_data["states"] if existing_data else None

            self.node_data[name] = {
                "name": name,
                "states": dialog.result["states"],
                "parents": parents,
                "cpt": dialog.result["cpt"]
            }

            if old_states != dialog.result["states"]:
                self.invalidate_descendants(name)

            self.statusBar().showMessage(
                f"Nodo {name} configurado correctamente"
            )
            QMessageBox.information(
                self, "Configuración guardada",
                f"Se guardaron los datos de {name}."
            )

    # ---------------- INVALIDAR DESCENDIENTES ----------------

    def invalidate_descendants(self, name):
        descendants = set()
        pending = [name]

        while pending:
            current = pending.pop()
            for origin, destination in self.connections:
                if origin == current and destination not in descendants:
                    descendants.add(destination)
                    pending.append(destination)

        for descendant in descendants:
            self.node_data.pop(descendant, None)

    # ---------------- CONECTAR NODOS ----------------

    def connect_nodes(self):
        if len(self.node_names) < 2:
            QMessageBox.warning(self, "Aviso", "Necesitas al menos dos nodos.")
            return

        parent, accepted = QInputDialog.getItem(
            self, "Nodo padre", "Selecciona el nodo que influye:",
            self.node_names, 0, False
        )
        if not accepted:
            return

        child_options = [name for name in self.node_names if name != parent]
        child, accepted = QInputDialog.getItem(
            self, "Nodo hijo", "Selecciona el nodo que recibe la influencia:",
            child_options, 0, False
        )
        if not accepted:
            return

        if (parent, child) in self.connections:
            QMessageBox.warning(self, "Aviso", "Esa conexión ya existe.")
            return
        if self.creates_cycle(parent, child):
            QMessageBox.warning(
                self, "Conexión no válida",
                "Esta conexión produciría un ciclo."
            )
            return

        self.connections.append((parent, child))
        self.node_data.pop(child, None)
        self.invalidate_descendants(child)
        self.draw_connections()
        self.statusBar().showMessage(f"Conexión creada: {parent} → {child}")

    # ---------------- COMPROBAR CICLOS ----------------

    def creates_cycle(self, parent, child):
        adjacency = {name: [] for name in self.node_names}
        for origin, destination in self.connections:
            adjacency[origin].append(destination)

        def has_path(current, target, visited):
            if current == target:
                return True
            visited.add(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited and has_path(neighbor, target, visited):
                    return True
            return False

        return has_path(child, parent, set())

    # ---------------- DIBUJAR CONEXIONES ----------------

    def draw_connections(self):
        for item in self.scene.items():
            if item.data(0) == "connection":
                self.scene.removeItem(item)

        for parent, child in self.connections:
            if parent not in self.node_positions or child not in self.node_positions:
                continue
            start = self.node_positions[parent]
            end = self.node_positions[child]
            dx, dy = end.x() - start.x(), end.y() - start.y()
            distance = math.hypot(dx, dy)
            if distance == 0:
                continue

            ux, uy = dx / distance, dy / distance
            start_point = QPointF(start.x() + ux * 50, start.y() + uy * 50)
            end_point = QPointF(end.x() - ux * 55, end.y() - uy * 55)
            connection_item = ConnectionItem(
                parent, child, start_point, end_point, ux, uy
            )
            self.scene.addItem(connection_item)

    # ---------------- DATOS DE RED ----------------

    def get_network_data(self):
        missing = [
            name for name in self.node_names if name not in self.node_data
        ]
        if missing:
            raise ValueError(
                "Configura primero estos nodos: " + ", ".join(missing)
            )

        nodes = []
        for name in self.node_names:
            data = self.node_data[name]
            nodes.append({
                "name": name,
                "states": data["states"],
                "parents": data["parents"],
                "cpt": data["cpt"]
            })
        return {"nodes": nodes}

    # ---------------- CONSULTA ----------------

    def run_query(self):
        if not self.node_names:
            QMessageBox.warning(self, "Red vacía", "Primero debes crear nodos.")
            return

        try:
            network = self.get_network_data()
            dialog = QueryDialog(self.node_data)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            query_variable, evidence = dialog.get_query()
            (result, elimination_order, original_order,
             comparison) = variable_elimination(
                network, [query_variable], evidence
            )

            results_dialog = ResultsDialog(
                query_variable, result, elimination_order,
                original_order, comparison
            )
            results_dialog.exec()

            self.statusBar().showMessage(
                f"Consulta calculada: {query_variable} (comparación de órdenes)"
            )

        except (ValueError, KeyError, TypeError) as error:
            QMessageBox.critical(self, "Error de consulta", str(error))

    # ---------------- GUARDAR JSON ----------------

    def save_network(self):
        if not self.node_names:
            QMessageBox.warning(self, "Red vacía", "No hay nodos para guardar.")
            return

        try:
            network = self.get_network_data()
            save_data = {
                "version": 1,
                "nodes": network["nodes"],
                "connections": [
                    {"parent": parent, "child": child}
                    for parent, child in self.connections
                ],
                "positions": {
                    name: {"x": position.x(), "y": position.y()}
                    for name, position in self.node_positions.items()
                }
            }

            file_path, _ = QFileDialog.getSaveFileName(
                self, "Guardar red bayesiana", "mi_red_bayesiana.json",
                "Archivos JSON (*.json)"
            )
            if not file_path:
                return
            if not file_path.lower().endswith(".json"):
                file_path += ".json"

            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(save_data, file, indent=4, ensure_ascii=False)

            QMessageBox.information(
                self, "Red guardada", "La red se guardó correctamente."
            )
            self.statusBar().showMessage("Red guardada correctamente")

        except (ValueError, KeyError, TypeError, OSError) as error:
            QMessageBox.critical(self, "Error al guardar", str(error))

    # ---------------- VALIDAR JSON ----------------

    def validate_loaded_network(self, data):
        if not isinstance(data, dict):
            raise ValueError("El archivo no contiene una red válida.")
        if "nodes" not in data or "connections" not in data:
            raise ValueError("El archivo debe contener nodos y conexiones.")

        nodes, connections = data["nodes"], data["connections"]
        if not isinstance(nodes, list) or not isinstance(connections, list):
            raise ValueError("La estructura de nodos o conexiones es inválida.")
        if not 1 <= len(nodes) <= 10:
            raise ValueError("La red debe tener entre 1 y 10 nodos.")

        names = []
        for node in nodes:
            if not isinstance(node, dict):
                raise ValueError("Hay un nodo con formato inválido.")
            name = node.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Hay un nombre de nodo inválido.")
            names.append(name)

        if len(names) != len(set(names)):
            raise ValueError("Hay nombres de nodos repetidos.")

        node_map = {node["name"]: node for node in nodes}

        # Validar nodos y CPT.
        for node in nodes:
            name = node["name"]
            states, parents, cpt = (
                node.get("states"), node.get("parents"), node.get("cpt")
            )

            if (
                not isinstance(states, list) or len(states) < 2
                or any(not isinstance(s, str) or not s for s in states)
                or len(states) != len(set(states))
            ):
                raise ValueError(f"Los estados de {name} son inválidos.")

            if not isinstance(parents, list) or not isinstance(cpt, list) or not cpt:
                raise ValueError(f"La estructura de {name} es inválida.")
            if len(parents) != len(set(parents)):
                raise ValueError(f"El nodo {name} tiene padres repetidos.")

            for parent in parents:
                if parent not in node_map or parent == name:
                    raise ValueError(f"El padre {parent} de {name} no es válido.")

            expected = list(product(*[
                node_map[parent]["states"] for parent in parents
            ]))
            if not parents:
                expected = [()]

            found = []
            for entry in cpt:
                if not isinstance(entry, dict):
                    raise ValueError(f"Una fila CPT de {name} es inválida.")

                probabilities = entry.get("probabilities")
                if not isinstance(probabilities, dict):
                    raise ValueError(f"Las probabilidades de {name} son inválidas.")
                if set(probabilities) != set(states):
                    raise ValueError(f"Los estados de la CPT de {name} no coinciden.")

                total = 0.0
                for state in states:
                    try:
                        probability = float(probabilities[state])
                    except (ValueError, TypeError):
                        raise ValueError(f"Probabilidad inválida en {name}.")
                    if not math.isfinite(probability) or not 0 <= probability <= 1:
                        raise ValueError(f"Probabilidad fuera de rango en {name}.")
                    total += probability

                if not math.isclose(total, 1.0, abs_tol=0.0001):
                    raise ValueError(f"Las probabilidades de {name} no suman 1.")

                if parents:
                    assignment = entry.get("parents")
                    if not isinstance(assignment, dict):
                        raise ValueError(f"Asignación de padres inválida en {name}.")
                    if set(assignment) != set(parents):
                        raise ValueError(f"Faltan padres en la CPT de {name}.")
                    combination = tuple(assignment[parent] for parent in parents)
                    if combination not in expected:
                        raise ValueError(f"Combinación de padres inválida en {name}.")
                    found.append(combination)
                else:
                    found.append(())

            if len(found) != len(expected) or set(found) != set(expected):
                raise ValueError(
                    f"La CPT de {name} está incompleta o tiene duplicados."
                )

        # Validar conexiones.
        checked_connections = []
        for connection in connections:
            if not isinstance(connection, dict):
                raise ValueError("Hay una conexión inválida.")
            parent, child = connection.get("parent"), connection.get("child")
            if parent not in node_map or child not in node_map:
                raise ValueError("Una conexión apunta a un nodo inexistente.")
            if parent == child:
                raise ValueError("Un nodo no puede conectarse consigo mismo.")
            if (parent, child) in checked_connections:
                raise ValueError("Hay conexiones duplicadas.")
            checked_connections.append((parent, child))

        for node in nodes:
            actual_parents = {
                parent for parent, child in checked_connections
                if child == node["name"]
            }
            if actual_parents != set(node["parents"]):
                raise ValueError(
                    f"Las conexiones y los padres de {node['name']} no coinciden."
                )

        # Comprobar ciclos.
        adjacency = {name: [] for name in names}
        for parent, child in checked_connections:
            adjacency[parent].append(child)

        visited, active = set(), set()

        def has_cycle(node):
            if node in active:
                return True
            if node in visited:
                return False
            visited.add(node)
            active.add(node)
            for neighbor in adjacency[node]:
                if has_cycle(neighbor):
                    return True
            active.remove(node)
            return False

        for name in names:
            if has_cycle(name):
                raise ValueError("El archivo contiene un ciclo en la red.")

        return True

    # ---------------- CARGAR JSON ----------------

    def load_network(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar red bayesiana", "", "Archivos JSON (*.json)"
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            self.validate_loaded_network(data)

            new_node_names = [node["name"] for node in data["nodes"]]
            new_node_data = {
                node["name"]: {
                    "name": node["name"],
                    "states": node["states"],
                    "parents": node["parents"],
                    "cpt": node["cpt"]
                }
                for node in data["nodes"]
            }
            new_connections = [
                (connection["parent"], connection["child"])
                for connection in data["connections"]
            ]

            saved_positions = data.get("positions", {})
            new_positions = {}

            for index, name in enumerate(new_node_names):
                if name in saved_positions:
                    position = saved_positions[name]
                    if not isinstance(position, dict):
                        raise ValueError(f"Posición inválida para {name}.")
                    x, y = float(position["x"]), float(position["y"])
                    if not math.isfinite(x) or not math.isfinite(y):
                        raise ValueError(f"Posición inválida para {name}.")
                else:
                    x = 100 + (index % 3) * 200
                    y = 100 + (index // 3) * 150
                new_positions[name] = QPointF(x, y)

            # Sustituir la red solo después de validar el archivo.
            self.scene.clear()
            self.deletion_history.clear()
            self.node_names = new_node_names
            self.node_data = new_node_data
            self.connections = new_connections
            self.node_positions = new_positions
            self.node_count = len(new_node_names)

            for name in self.node_names:
                position = self.node_positions[name]
                x, y = position.x() - 50, position.y() - 50
                node_item = NodeItem(x, y, name, self)
                self.scene.addItem(node_item)

            self.draw_connections()
            self.update_scene_bounds()
            self.statusBar().showMessage(
                f"Red cargada: {self.node_count} nodos"
            )
            QMessageBox.information(
                self, "Carga completada",
                "La red bayesiana se cargó correctamente."
            )

        except (
            OSError, json.JSONDecodeError, ValueError,
            KeyError, TypeError, AttributeError
        ) as error:
            QMessageBox.critical(self, "Error al cargar", str(error))

    # ---------------- ELIMINAR Y DESHACER ----------------

    def _capture_network_state(self):
        """Guarda una copia completa del estado antes de una eliminación."""
        positions = {}
        for name, position in self.node_positions.items():
            positions[name] = (position.x(), position.y())
        return {
            "node_names": list(self.node_names),
            "connections": list(self.connections),
            "node_positions": positions,
            "node_data": copy.deepcopy(self.node_data),
        }

    def _restore_network_state(self, state):
        """Reconstruye la escena y los datos desde una copia del estado."""
        self.node_names = list(state["node_names"])
        self.connections = list(state["connections"])
        self.node_data = copy.deepcopy(state["node_data"])
        self.node_positions = {
            name: QPointF(x, y)
            for name, (x, y) in state["node_positions"].items()
        }
        self.node_count = len(self.node_names)

        self.scene.clear()
        for name in self.node_names:
            center = self.node_positions[name]
            node_item = NodeItem(center.x() - 50, center.y() - 50, name, self)
            self.scene.addItem(node_item)

        self.draw_connections()
        if self.node_names:
            self.update_scene_bounds()
        else:
            self.scene.setSceneRect(0, 0, 1000, 700)
        self.statusBar().showMessage("Última eliminación deshecha. Se restauró la red anterior.")

    def undo_last_deletion(self):
        """Deshace la última operación de eliminación."""
        if not self.deletion_history:
            self.statusBar().showMessage("No hay eliminaciones que se puedan deshacer.")
            return
        previous_state = self.deletion_history.pop()
        self._restore_network_state(previous_state)

    def delete_selected_items(self):
        """Elimina nodos, conexiones o ambos según la selección actual."""
        selected_items = self.scene.selectedItems()
        selected_nodes = [
            item for item in selected_items
            if isinstance(item, NodeItem) and item.data(0) == "node"
        ]
        selected_edges = [
            item for item in selected_items
            if item.data(0) == "connection"
        ]

        if not selected_nodes and not selected_edges:
            self.statusBar().showMessage(
                "Selecciona uno o varios nodos o flechas para eliminarlos."
            )
            return

        # Una sola captura permite deshacer toda la eliminación conjunta.
        self.deletion_history.append(self._capture_network_state())
        if len(self.deletion_history) > self.max_undo_steps:
            self.deletion_history.pop(0)

        removed_node_names = {item.name for item in selected_nodes}
        explicitly_removed_edges = {
            tuple(item.data(1)) for item in selected_edges
        }

        # Las CPT se reinician solo en los hijos directos afectados que
        # permanezcan en la red. Los descendientes más lejanos se conservan.
        affected_children = set()
        for parent, child in self.connections:
            if parent in removed_node_names and child not in removed_node_names:
                affected_children.add(child)
            if (parent, child) in explicitly_removed_edges and child not in removed_node_names:
                affected_children.add(child)

        for child in affected_children:
            self.node_data.pop(child, None)

        self.connections = [
            (parent, child)
            for parent, child in self.connections
            if parent not in removed_node_names
            and child not in removed_node_names
            and (parent, child) not in explicitly_removed_edges
        ]

        for name in removed_node_names:
            self.node_data.pop(name, None)
            self.node_positions.pop(name, None)
            if name in self.node_names:
                self.node_names.remove(name)

        self.node_count = len(self.node_names)
        self.scene.clear()
        for name in self.node_names:
            center = self.node_positions[name]
            node_item = NodeItem(center.x() - 50, center.y() - 50, name, self)
            self.scene.addItem(node_item)
        self.draw_connections()

        if self.node_names:
            self.update_scene_bounds()
        else:
            self.scene.setSceneRect(0, 0, 1000, 700)

        details = []
        if removed_node_names:
            details.append("nodos: " + ", ".join(sorted(removed_node_names)))
        removed_edges = explicitly_removed_edges | {
            edge for edge in self.deletion_history[-1]["connections"]
            if edge[0] in removed_node_names or edge[1] in removed_node_names
        }
        if removed_edges:
            details.append(f"{len(removed_edges)} conexión(es)")
        if affected_children:
            details.append("CPT reiniciadas: " + ", ".join(sorted(affected_children)))
        self.statusBar().showMessage(
            "Eliminado: " + "; ".join(details) + ". Ctrl+Z para deshacer."
        )

    # ---------------- LIMPIAR RED ----------------

    def clear_network(self):
        confirm = QMessageBox.question(
            self, "Limpiar red", "¿Deseas eliminar todos los nodos?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if confirm == QMessageBox.StandardButton.Yes:
            self.scene.clear()
            self.node_count = 0
            self.node_names.clear()
            self.connections.clear()
            self.node_positions.clear()
            self.node_data.clear()
            self.deletion_history.clear()
            self.scene.setSceneRect(0, 0, 1000, 700)
            self.statusBar().showMessage("Red limpiada correctamente")


# ==================================================
# 9. EJECUCIÓN
# ==================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
