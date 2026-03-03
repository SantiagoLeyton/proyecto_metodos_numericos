from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from funciones.definiciones import get_exercises
from metodos.bisection import bisection
from utils.validaciones import (
    format_number,
    validate_bracket_sign_change,
    validate_max_iterations,
    validate_tolerance,
)


# =========================================================
# ESTRUCTURA AUXILIAR PARA RESULTADOS FINALES
# =========================================================

@dataclass
class FinalResultWidgets:
    root_value_var: tk.StringVar
    iterations_var: tk.StringVar
    final_error_var: tk.StringVar
    time_var: tk.StringVar
    status_var: tk.StringVar


# =========================================================
# TAB BASE REUTILIZABLE
# =========================================================

class BaseMethodTab(ttk.Frame):

    def __init__(
        self,
        master: ttk.Notebook,
        columns: List[str],
    ) -> None:
        super().__init__(master)

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # -------- INPUT PANEL --------
        self.input_frame = ttk.LabelFrame(self, text="Panel de Entrada")
        self.input_frame.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

        # -------- CONTENT --------
        self.content_frame = ttk.Frame(self)
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.rowconfigure(0, weight=1)
        self.content_frame.rowconfigure(1, weight=1)
        self.content_frame.rowconfigure(2, weight=0)

        # -------- TABLE --------
        self.table_frame = ttk.LabelFrame(self.content_frame, text="Tabla de Resultados")
        self.table_frame.grid(row=0, column=0, sticky="nsew", pady=5)

        self.tree = ttk.Treeview(self.table_frame, columns=columns, show="headings")
        self.tree.pack(fill="both", expand=True)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="center")

        # -------- PLOTS --------
        self.plot_frame = ttk.LabelFrame(self.content_frame, text="Gráficas")
        self.plot_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.columnconfigure(1, weight=1)

        # Plot función
        self.figure_func = Figure(figsize=(5, 4), dpi=100)
        self.ax_func = self.figure_func.add_subplot(111)
        self.canvas_func = FigureCanvasTkAgg(self.figure_func, master=self.plot_frame)
        self.canvas_func.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Plot error
        self.figure_err = Figure(figsize=(5, 4), dpi=100)
        self.ax_err = self.figure_err.add_subplot(111)
        self.canvas_err = FigureCanvasTkAgg(self.figure_err, master=self.plot_frame)
        self.canvas_err.get_tk_widget().grid(row=0, column=1, sticky="nsew")

        # -------- FINAL PANEL --------
        self.final_frame = ttk.LabelFrame(self.content_frame, text="Resultados Finales")
        self.final_frame.grid(row=2, column=0, sticky="ew", pady=5)

        self.root_value_var = tk.StringVar(value="-")
        self.iterations_var = tk.StringVar(value="-")
        self.final_error_var = tk.StringVar(value="-")
        self.time_var = tk.StringVar(value="-")
        self.status_var = tk.StringVar(value="-")

        labels = [
            ("Raíz aproximada:", self.root_value_var),
            ("Iteraciones:", self.iterations_var),
            ("Error final:", self.final_error_var),
            ("Tiempo (s):", self.time_var),
            ("Estado:", self.status_var),
        ]

        for i, (text, var) in enumerate(labels):
            ttk.Label(self.final_frame, text=text).grid(row=i, column=0, sticky="w")
            ttk.Label(self.final_frame, textvariable=var).grid(row=i, column=1, sticky="w")

        # -------- BUTTONS --------
        btn_frame = ttk.Frame(self.input_frame)
        btn_frame.grid(row=99, column=0, columnspan=2, pady=10)

        self.calculate_btn = ttk.Button(btn_frame, text="Calcular")
        self.calculate_btn.grid(row=0, column=0, padx=5)

        self.clear_btn = ttk.Button(btn_frame, text="Limpiar", command=self.clear_all)
        self.clear_btn.grid(row=0, column=1, padx=5)

    def clear_all(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.ax_func.clear()
        self.ax_err.clear()
        self.canvas_func.draw()
        self.canvas_err.draw()

        self.root_value_var.set("-")
        self.iterations_var.set("-")
        self.final_error_var.set("-")
        self.time_var.set("-")
        self.status_var.set("-")


# =========================================================
# APP PRINCIPAL
# =========================================================

class NumericMethodsApp(tk.Tk):

    def __init__(self) -> None:
        super().__init__()
        self.title("Proyecto Métodos Numéricos")
        self.geometry("1400x800")

        self.exercises = get_exercises()

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self._build_tabs()

    # =====================================================
    # CONSTRUCCIÓN DE TABS
    # =====================================================

    def _build_tabs(self):

        # ---- EJERCICIO 1 ----
        ex1_columns = ["n", "a", "b", "c", "f(c)", "Error abs", "Error rel (%)"]

        self.tab_ex1 = BaseMethodTab(self.notebook, ex1_columns)
        self.notebook.add(self.tab_ex1, text="Ej.1 Bisección")

        self._build_ex1_inputs()
        self.tab_ex1.calculate_btn.config(command=self._calculate_ex1)

    # =====================================================
    # INPUTS EJ1
    # =====================================================

    def _add_entry(self, parent, label, default, row):
        var = tk.StringVar(value=default)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=var, width=15).grid(row=row, column=1)
        return var

    def _build_ex1_inputs(self):
        self.a_var = self._add_entry(self.tab_ex1.input_frame, "a:", "0.5", 0)
        self.b_var = self._add_entry(self.tab_ex1.input_frame, "b:", "2.5", 1)
        self.tol_var = self._add_entry(self.tab_ex1.input_frame, "Tolerancia:", "1e-6", 2)
        self.max_iter_var = self._add_entry(self.tab_ex1.input_frame, "Max iter:", "100", 3)

    # =====================================================
    # CÁLCULO EJ1
    # =====================================================

    def _calculate_ex1(self):

        tab = self.tab_ex1
        tab.clear_all()

        try:
            a = float(self.a_var.get())
            b = float(self.b_var.get())
            tol = float(self.tol_var.get())
            max_iter = int(self.max_iter_var.get())
        except ValueError:
            messagebox.showerror("Error", "Entradas inválidas.")
            return

        f = self.exercises["EX1"].f

        # Validaciones
        if not validate_tolerance(tol).ok:
            messagebox.showerror("Error", "Tolerancia inválida.")
            return

        if not validate_max_iterations(max_iter).ok:
            messagebox.showerror("Error", "Max iter inválido.")
            return

        if not validate_bracket_sign_change(f, a, b).ok:
            messagebox.showerror("Error", "El intervalo no contiene raíz.")
            return

        result = bisection(f, a, b, tol, max_iter)

        # ---- Tabla ----
        for row in result.rows:
            tab.tree.insert(
                "",
                "end",
                values=(
                    row.n,
                    format_number(row.a),
                    format_number(row.b),
                    format_number(row.c),
                    format_number(row.fc),
                    format_number(row.abs_error),
                    format_number(row.rel_error_pct),
                ),
            )

        # ---- Panel Final ----
        tab.root_value_var.set(format_number(result.root))
        tab.iterations_var.set(str(result.iterations))
        tab.final_error_var.set(format_number(result.final_abs_error))
        tab.time_var.set(format_number(result.elapsed_seconds))
        tab.status_var.set(result.message)

        # ---- Plot Función ----
        tab.ax_func.clear()
        xs = np.linspace(a - 1, b + 1, 400)
        ys = [f(x) for x in xs]

        tab.ax_func.plot(xs, ys, label="T(λ)")
        tab.ax_func.axhline(0)

        c_vals = result.c_values
        c_ys = [f(c) for c in c_vals]

        tab.ax_func.plot(c_vals, c_ys, "o", label="Iteraciones")
        tab.ax_func.legend()
        tab.canvas_func.draw()

        # ---- Plot Error ----
        tab.ax_err.clear()
        if result.abs_errors:
            iters = range(2, 2 + len(result.abs_errors))
            tab.ax_err.semilogy(iters, result.abs_errors, "o-")

        tab.ax_err.set_title("Error absoluto (log)")
        tab.canvas_err.draw()


# =========================================================
# FUNCIÓN PARA CREAR APP
# =========================================================

def create_app() -> NumericMethodsApp:
    return NumericMethodsApp()