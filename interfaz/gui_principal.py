from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from funciones.definiciones import get_exercises
from utils.validaciones import (
    format_number,
    validate_bracket_sign_change,
    validate_max_iterations,
    validate_tolerance,
)

from metodos.bisection import bisection
from metodos.false_position import false_position
from metodos.fixed_point import fixed_point
from metodos.newton import newton
from metodos.secant import secant


@dataclass
class FinalPanelVars:
    root: tk.StringVar
    iterations: tk.StringVar
    final_error: tk.StringVar
    elapsed: tk.StringVar
    status: tk.StringVar


class MethodTab(ttk.Frame):
    """Tab con: entradas, tabla, plots, panel final."""

    def __init__(self, master: ttk.Notebook, columns: List[str], title: str) -> None:
        super().__init__(master)
        self.title = title

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Izquierda: inputs
        self.input_frame = ttk.LabelFrame(self, text="Panel de Entrada")
        self.input_frame.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

        # Derecha: tabla + plots + final
        self.right_frame = ttk.Frame(self)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.right_frame.columnconfigure(0, weight=1)
        self.right_frame.rowconfigure(0, weight=1)
        self.right_frame.rowconfigure(1, weight=1)
        self.right_frame.rowconfigure(2, weight=0)

        # Tabla
        self.table_frame = ttk.LabelFrame(self.right_frame, text="Tabla de Resultados")
        self.table_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        self.table_frame.columnconfigure(0, weight=1)
        self.table_frame.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(self.table_frame, columns=columns, show="headings")
        self.tree.grid(row=0, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=vsb.set)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="center")

        # Plots
        self.plots_frame = ttk.LabelFrame(self.right_frame, text="Gráficas")
        self.plots_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        self.plots_frame.columnconfigure(0, weight=1)
        self.plots_frame.columnconfigure(1, weight=1)
        self.plots_frame.rowconfigure(0, weight=1)

        self.figure_func = Figure(figsize=(5, 4), dpi=100)
        self.ax_func = self.figure_func.add_subplot(111)
        self.canvas_func = FigureCanvasTkAgg(self.figure_func, master=self.plots_frame)
        self.canvas_func.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.figure_err = Figure(figsize=(5, 4), dpi=100)
        self.ax_err = self.figure_err.add_subplot(111)
        self.canvas_err = FigureCanvasTkAgg(self.figure_err, master=self.plots_frame)
        self.canvas_err.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # Panel final
        self.final_frame = ttk.LabelFrame(self.right_frame, text="Resultados Finales")
        self.final_frame.grid(row=2, column=0, sticky="ew")

        self.final_vars = FinalPanelVars(
            root=tk.StringVar(value="-"),
            iterations=tk.StringVar(value="-"),
            final_error=tk.StringVar(value="-"),
            elapsed=tk.StringVar(value="-"),
            status=tk.StringVar(value="-"),
        )

        labels = [
            ("Raíz aproximada:", self.final_vars.root),
            ("Iteraciones:", self.final_vars.iterations),
            ("Error final:", self.final_vars.final_error),
            ("Tiempo (s):", self.final_vars.elapsed),
            ("Estado:", self.final_vars.status),
        ]

        for i, (txt, var) in enumerate(labels):
            ttk.Label(self.final_frame, text=txt).grid(row=i, column=0, sticky="w", padx=8, pady=2)
            ttk.Label(self.final_frame, textvariable=var).grid(row=i, column=1, sticky="w", padx=8, pady=2)

        self.final_frame.columnconfigure(1, weight=1)

        # Botones comunes
        btn_frame = ttk.Frame(self.input_frame)
        btn_frame.grid(row=999, column=0, columnspan=3, pady=10, sticky="ew")

        self.btn_calc = ttk.Button(btn_frame, text="Calcular")
        self.btn_calc.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.btn_clear = ttk.Button(btn_frame, text="Limpiar", command=self.clear_all)
        self.btn_clear.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

    def clear_all(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.ax_func.clear()
        self.ax_err.clear()
        self.canvas_func.draw()
        self.canvas_err.draw()

        self.final_vars.root.set("-")
        self.final_vars.iterations.set("-")
        self.final_vars.final_error.set("-")
        self.final_vars.elapsed.set("-")
        self.final_vars.status.set("-")


class NumericMethodsApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Proyecto Métodos Numéricos - Tabs")
        self.geometry("1500x850")

        self.exercises = get_exercises()

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.tabs: Dict[str, MethodTab] = {}
        self._build_tabs()

    # --------------------------
    # Helpers UI
    # --------------------------
    def _add_entry(self, parent: ttk.Frame, label: str, default: str, row: int, col: int = 0) -> tk.StringVar:
        var = tk.StringVar(value=default)
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=6, pady=4)
        ttk.Entry(parent, textvariable=var, width=18).grid(row=row, column=col + 1, sticky="ew", padx=6, pady=4)
        parent.columnconfigure(col + 1, weight=1)
        return var

    def _add_combobox(self, parent: ttk.Frame, label: str, values: List[str], default: str, row: int) -> tk.StringVar:
        var = tk.StringVar(value=default)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        cb = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=18)
        cb.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        parent.columnconfigure(1, weight=1)
        return var

    # --------------------------
    # Tabs
    # --------------------------
    def _build_tabs(self) -> None:
        # EJ1
        ex1_cols = ["n", "a", "b", "c", "f(c)", "Error abs", "Error rel (%)"]
        tab1 = MethodTab(self.notebook, ex1_cols, "Ejercicio 1 - Bisección")
        self.notebook.add(tab1, text="Ej.1 Bisección")
        self.tabs["EX1"] = tab1
        self._build_ex1(tab1)

        # EJ2 Comparación
        ex2_cols = ["Método", "n", "x_n", "f(x_n)", "Error abs", "Error rel (%)"]
        tab2 = MethodTab(self.notebook, ex2_cols, "Ejercicio 2 - Comparación")
        self.notebook.add(tab2, text="Ej.2 Comparación")
        self.tabs["EX2"] = tab2
        self._build_ex2(tab2)

        # EJ3 Punto fijo
        ex3_cols = ["x0", "n", "x_n", "g(x_n)", "|x_n-g(x_n)|", "Error abs", "Error rel (%)"]
        tab3 = MethodTab(self.notebook, ex3_cols, "Ejercicio 3 - Punto Fijo")
        self.notebook.add(tab3, text="Ej.3 Punto Fijo")
        self.tabs["EX3"] = tab3
        self._build_ex3(tab3)

        # EJ4 Newton
        ex4_cols = ["n", "x_n", "f(x_n)", "f'(x_n)", "Error abs", "Error rel (%)"]
        tab4 = MethodTab(self.notebook, ex4_cols, "Ejercicio 4 - Newton")
        self.notebook.add(tab4, text="Ej.4 Newton")
        self.tabs["EX4"] = tab4
        self._build_ex4(tab4)

        # EJ5 Secante
        ex5_cols = ["n", "x_{n-1}", "x_n", "f(x_{n-1})", "f(x_n)", "x_{n+1}", "Error abs"]
        tab5 = MethodTab(self.notebook, ex5_cols, "Ejercicio 5 - Secante")
        self.notebook.add(tab5, text="Ej.5 Secante")
        self.tabs["EX5"] = tab5
        self._build_ex5(tab5)

        # EJ5 Comparación
        ex5c_cols = ["Método", "Iteraciones", "Eval f", "Eval f'", "Tiempo (s)"]
        tab6 = MethodTab(self.notebook, ex5c_cols, "Ejercicio 5 - Comparación")
        self.notebook.add(tab6, text="Ej.5 Comparación")
        self.tabs["EX5C"] = tab6
        self._build_ex5_compare(tab6)

    # --------------------------
    # EJ1
    # --------------------------
    def _build_ex1(self, tab: MethodTab) -> None:
        tab.a = self._add_entry(tab.input_frame, "a:", "0.5", 0)
        tab.b = self._add_entry(tab.input_frame, "b:", "2.5", 1)
        tab.tol = self._add_entry(tab.input_frame, "Tolerancia:", "1e-6", 2)
        tab.max_iter = self._add_entry(tab.input_frame, "Max iter:", "100", 3)

        ttk.Label(
            tab.input_frame,
            text="Función: T(λ)=2.5+0.8λ²-3.2λ+ln(λ+1)\nRequiere cambio de signo.",
            wraplength=260,
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=6, pady=(10, 4))

        tab.btn_calc.configure(command=lambda: self._calculate_ex1(tab))

    def _calculate_ex1(self, tab: MethodTab) -> None:
        tab.clear_all()

        try:
            a = float(tab.a.get())
            b = float(tab.b.get())
            tol = float(tab.tol.get())
            max_iter = int(float(tab.max_iter.get()))
        except ValueError as exc:
            messagebox.showerror("Error", f"Entradas inválidas: {exc}")
            return

        if not validate_tolerance(tol).ok:
            messagebox.showerror("Error", "Tolerancia inválida.")
            return
        if not validate_max_iterations(max_iter).ok:
            messagebox.showerror("Error", "Max iter inválido.")
            return

        f = self.exercises["EX1"].f

        try:
            v = validate_bracket_sign_change(f, a, b)
        except Exception as exc:
            messagebox.showerror("Error", f"Error evaluando la función:\n{exc}")
            return
        if not v.ok:
            # Esto es CORRECTO según las specs
            tab.final_vars.status.set(v.message)
            messagebox.showerror("Validación", v.message)
            return

        res = bisection(f, a, b, tol, max_iter)

        for r in res.rows:
            tab.tree.insert("", "end", values=(
                r.n,
                format_number(r.a),
                format_number(r.b),
                format_number(r.c),
                format_number(r.fc),
                format_number(r.abs_error),
                format_number(r.rel_error_pct),
            ))

        tab.final_vars.root.set(format_number(res.root))
        tab.final_vars.iterations.set(str(res.iterations))
        tab.final_vars.final_error.set(format_number(res.final_abs_error))
        tab.final_vars.elapsed.set(format_number(res.elapsed_seconds))
        tab.final_vars.status.set(res.message + f" | Eval f: {res.func_evals}")

        # Plot función
        tab.ax_func.clear()
        left = min(a, b)
        right = max(a, b)
        pad = (right - left) * 0.2
        xs = np.linspace(left - pad, right + pad, 500)
        ys = []
        for x in xs:
            try:
                ys.append(f(float(x)))
            except Exception:
                ys.append(np.nan)

        tab.ax_func.plot(xs, ys, label="T(λ)")
        tab.ax_func.axhline(0.0)
        # puntos iterativos
        c_vals = res.c_values
        c_ys = [f(c) for c in c_vals]
        tab.ax_func.plot(c_vals, c_ys, marker="o", linestyle="None", label="Iteraciones")
        tab.ax_func.plot([res.root], [f(res.root)], marker="*", markersize=12, linestyle="None", label="Raíz final")
        tab.ax_func.set_title("Función y convergencia (Bisección)")
        tab.ax_func.set_xlabel("λ")
        tab.ax_func.set_ylabel("T(λ)")
        tab.ax_func.legend()
        tab.canvas_func.draw()

        # Plot error log
        tab.ax_err.clear()
        tab.ax_err.set_title("Convergencia del error (log)")
        tab.ax_err.set_xlabel("Iteración")
        tab.ax_err.set_ylabel("Error absoluto")
        if res.abs_errors:
            iters = list(range(2, 2 + len(res.abs_errors)))
            tab.ax_err.semilogy(iters, res.abs_errors, marker="o")
        tab.canvas_err.draw()

    # --------------------------
    # EJ2
    # --------------------------
    def _build_ex2(self, tab: MethodTab) -> None:
        tab.a = self._add_entry(tab.input_frame, "a:", "2.0", 0)
        tab.b = self._add_entry(tab.input_frame, "b:", "4.0", 1)
        tab.tol = self._add_entry(tab.input_frame, "Tolerancia:", "1e-7", 2)
        tab.max_iter = self._add_entry(tab.input_frame, "Max iter:", "200", 3)

        ttk.Label(
            tab.input_frame,
            text="Comparación: Bisección vs Falsa Posición\nE(x)=x^3-6x^2+11x-6.5",
            wraplength=260,
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=6, pady=(10, 4))

        tab.btn_calc.configure(command=lambda: self._calculate_ex2_compare(tab))

    def _calculate_ex2_compare(self, tab: MethodTab) -> None:
        tab.clear_all()

        try:
            a = float(tab.a.get())
            b = float(tab.b.get())
            tol = float(tab.tol.get())
            max_iter = int(float(tab.max_iter.get()))
        except ValueError as exc:
            messagebox.showerror("Error", f"Entradas inválidas: {exc}")
            return

        if not validate_tolerance(tol).ok or not validate_max_iterations(max_iter).ok:
            messagebox.showerror("Error", "Tolerancia o max iter inválidos.")
            return

        f = self.exercises["EX2"].f
        try:
            v = validate_bracket_sign_change(f, a, b)
        except Exception as exc:
            messagebox.showerror("Error", f"Error evaluando f:\n{exc}")
            return
        if not v.ok:
            messagebox.showerror("Validación", v.message)
            tab.final_vars.status.set(v.message)
            return

        res_bis = bisection(f, a, b, tol, max_iter)
        res_fp = false_position(f, a, b, tol, max_iter)

        # Tabla comparativa (mezclada)
        for r in res_bis.rows:
            tab.tree.insert("", "end", values=(
                "Bisección",
                r.n,
                format_number(r.c),
                format_number(r.fc),
                format_number(r.abs_error),
                format_number(r.rel_error_pct),
            ))

        for r in res_fp.rows:
            tab.tree.insert("", "end", values=(
                "Falsa Posición",
                r.n,
                format_number(r.c),
                format_number(r.fc),
                format_number(r.abs_error),
                format_number(r.rel_error_pct),
            ))

        # Panel final: resumen rápido
        tab.final_vars.root.set(f"Bis: {format_number(res_bis.root)} | FP: {format_number(res_fp.root)}")
        tab.final_vars.iterations.set(f"Bis: {res_bis.iterations} | FP: {res_fp.iterations}")
        tab.final_vars.final_error.set(f"Bis: {format_number(res_bis.final_abs_error)} | FP: {format_number(res_fp.final_abs_error)}")
        tab.final_vars.elapsed.set(f"Bis: {format_number(res_bis.elapsed_seconds)} | FP: {format_number(res_fp.elapsed_seconds)}")
        tab.final_vars.status.set(
            f"Eval f -> Bis: {res_bis.func_evals} | FP: {res_fp.func_evals}"
        )

        # Plot función + puntos finales
        tab.ax_func.clear()
        xs = np.linspace(min(a, b) - 0.5, max(a, b) + 0.5, 600)
        ys = [f(float(x)) for x in xs]
        tab.ax_func.plot(xs, ys, label="E(x)")
        tab.ax_func.axhline(0.0)
        tab.ax_func.plot(res_bis.c_values, [f(x) for x in res_bis.c_values], marker="o", linestyle="None", label="Iter. Bisección")
        tab.ax_func.plot(res_fp.c_values, [f(x) for x in res_fp.c_values], marker="x", linestyle="None", label="Iter. Falsa Posición")
        tab.ax_func.set_title("Función y convergencia (comparación)")
        tab.ax_func.legend()
        tab.canvas_func.draw()

        # Plot error log superpuesto
        tab.ax_err.clear()
        tab.ax_err.set_title("Error absoluto (log) - comparación")
        tab.ax_err.set_xlabel("Iteración")
        tab.ax_err.set_ylabel("Error abs")

        if res_bis.abs_errors:
            it_b = list(range(2, 2 + len(res_bis.abs_errors)))
            tab.ax_err.semilogy(it_b, res_bis.abs_errors, marker="o", label="Bisección")
        if res_fp.abs_errors:
            it_f = list(range(2, 2 + len(res_fp.abs_errors)))
            tab.ax_err.semilogy(it_f, res_fp.abs_errors, marker="x", label="Falsa Posición")

        tab.ax_err.legend()
        tab.canvas_err.draw()

    # --------------------------
    # EJ3
    # --------------------------
    def _build_ex3(self, tab: MethodTab) -> None:
        tab.x0_choice = self._add_combobox(tab.input_frame, "x0:", ["0.5", "1.0", "1.5", "2.0"], "1.0", 0)
        tab.tol = self._add_entry(tab.input_frame, "Tolerancia:", "1e-8", 1)
        tab.max_iter = self._add_entry(tab.input_frame, "Max iter:", "200", 2)

        ttk.Label(
            tab.input_frame,
            text="g(x)=0.5cos(x)+1.5\nSe grafica cobweb + comparación x0.",
            wraplength=260,
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=(10, 4))

        tab.btn_calc.configure(command=lambda: self._calculate_ex3(tab))

    def _calculate_ex3(self, tab: MethodTab) -> None:
        tab.clear_all()

        try:
            tol = float(tab.tol.get())
            max_iter = int(float(tab.max_iter.get()))
            x0_selected = float(tab.x0_choice.get())
        except ValueError as exc:
            messagebox.showerror("Error", f"Entradas inválidas: {exc}")
            return

        if not validate_tolerance(tol).ok or not validate_max_iterations(max_iter).ok:
            messagebox.showerror("Error", "Tolerancia o max iter inválidos.")
            return

        ex = self.exercises["EX3"]
        if ex.g is None:
            messagebox.showerror("Error", "No existe g(x) para este ejercicio.")
            return

        g = ex.g
        gprime = ex.g_prime

        # correr todos para comparación
        initial_values = [0.5, 1.0, 1.5, 2.0]
        results = {}
        for x0 in initial_values:
            results[x0] = fixed_point(g=g, x0=x0, tol=tol, max_iter=max_iter, g_prime=gprime)

        # mostrar en tabla solo el x0 seleccionado
        res = results[x0_selected]
        for r in res.rows:
            tab.tree.insert("", "end", values=(
                format_number(x0_selected),
                r.n,
                format_number(r.x_n),
                format_number(r.g_xn),
                format_number(r.residual),
                format_number(r.abs_error),
                format_number(r.rel_error_pct),
            ))

        # panel final
        gprime_msg = ""
        if res.gprime_at_start is not None:
            gprime_msg = f" | g'(x0)={format_number(res.gprime_at_start)}"
        tab.final_vars.root.set(format_number(res.root))
        tab.final_vars.iterations.set(str(res.iterations))
        tab.final_vars.final_error.set(format_number(res.final_abs_error))
        tab.final_vars.elapsed.set(format_number(res.elapsed_seconds))
        tab.final_vars.status.set(res.message + gprime_msg + f" | Eval g: {res.func_evals}")

        # Plot cobweb (izquierda)
        tab.ax_func.clear()
        # dominio
        x_min, x_max = 0.0, 3.0
        xs = np.linspace(x_min, x_max, 400)
        ys_g = [g(float(x)) for x in xs]
        tab.ax_func.plot(xs, xs, label="y = x")
        tab.ax_func.plot(xs, ys_g, label="y = g(x)")

        # cobweb para x0 seleccionado
        x_n = x0_selected
        for _ in range(min(res.iterations, 50)):
            y_n = g(x_n)
            # vertical: (x_n, x_n) -> (x_n, g(x_n))
            tab.ax_func.plot([x_n, x_n], [x_n, y_n])
            # horizontal: (x_n, g(x_n)) -> (g(x_n), g(x_n))
            tab.ax_func.plot([x_n, y_n], [y_n, y_n])
            x_n = y_n

        tab.ax_func.set_title("Cobweb (Punto Fijo)")
        tab.ax_func.set_xlabel("x")
        tab.ax_func.set_ylabel("y")
        tab.ax_func.legend()
        tab.canvas_func.draw()

        # Plot error comparación (derecha)
        tab.ax_err.clear()
        tab.ax_err.set_title("Error abs (log) - comparación x0")
        tab.ax_err.set_xlabel("Iteración")
        tab.ax_err.set_ylabel("Error abs")
        for x0, rr in results.items():
            if rr.abs_errors:
                it = list(range(1, 1 + len(rr.abs_errors)))
                tab.ax_err.semilogy(it, rr.abs_errors, marker="o", linestyle="-", label=f"x0={x0}")
        tab.ax_err.legend()
        tab.canvas_err.draw()

    # --------------------------
    # EJ4
    # --------------------------
    def _build_ex4(self, tab: MethodTab) -> None:
        tab.x0_choice = self._add_combobox(tab.input_frame, "x0:", ["1.0", "2.0", "3.0", "5.0"], "3.0", 0)
        tab.tol = self._add_entry(tab.input_frame, "Tolerancia:", "1e-10", 1)
        tab.max_iter = self._add_entry(tab.input_frame, "Max iter:", "200", 2)

        ttk.Label(
            tab.input_frame,
            text="Newton-Raphson\nSe dibujan tangentes + error log.",
            wraplength=260,
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=(10, 4))

        tab.btn_calc.configure(command=lambda: self._calculate_ex4(tab))

    def _calculate_ex4(self, tab: MethodTab) -> None:
        tab.clear_all()

        try:
            x0 = float(tab.x0_choice.get())
            tol = float(tab.tol.get())
            max_iter = int(float(tab.max_iter.get()))
        except ValueError as exc:
            messagebox.showerror("Error", f"Entradas inválidas: {exc}")
            return

        if not validate_tolerance(tol).ok or not validate_max_iterations(max_iter).ok:
            messagebox.showerror("Error", "Tolerancia o max iter inválidos.")
            return

        ex = self.exercises["EX4"]
        if ex.f_prime is None:
            messagebox.showerror("Error", "No existe f'(x) para Newton en EX4.")
            return

        f = ex.f
        fp = ex.f_prime

        res = newton(f=f, f_prime=fp, x0=x0, tol=tol, max_iter=max_iter)

        for r in res.rows:
            tab.tree.insert("", "end", values=(
                r.n,
                format_number(r.x_n),
                format_number(r.fx),
                format_number(r.fpx),
                format_number(r.abs_error),
                format_number(r.rel_error_pct),
            ))

        tab.final_vars.root.set(format_number(res.root))
        tab.final_vars.iterations.set(str(res.iterations))
        tab.final_vars.final_error.set(format_number(res.final_abs_error))
        tab.final_vars.elapsed.set(format_number(res.elapsed_seconds))
        tab.final_vars.status.set(res.message + f" | Eval f: {res.func_evals} | Eval f': {res.deriv_evals}")

        # Plot función + tangentes
        tab.ax_func.clear()
        x_min, x_max = 0.0, 6.0
        xs = np.linspace(x_min, x_max, 600)
        ys = [f(float(x)) for x in xs]
        tab.ax_func.plot(xs, ys, label="T(n)")
        tab.ax_func.axhline(0.0)

        # dibujar tangentes (limitadas)
        for tng in res.tangents[:10]:
            # y = y0 + m(x - x0)
            x_line = np.linspace(x_min, x_max, 2)
            y_line = tng.y0 + tng.slope * (x_line - tng.x0)
            tab.ax_func.plot(x_line, y_line, linestyle="--", alpha=0.7)

        tab.ax_func.set_title("Newton: función y tangentes")
        tab.ax_func.legend()
        tab.canvas_func.draw()

        # Error log
        tab.ax_err.clear()
        tab.ax_err.set_title("Error absoluto (log) - Newton")
        tab.ax_err.set_xlabel("Iteración")
        tab.ax_err.set_ylabel("Error abs")
        if res.abs_errors:
            it = list(range(1, 1 + len(res.abs_errors)))
            tab.ax_err.semilogy(it, res.abs_errors, marker="o")
        tab.canvas_err.draw()

    # --------------------------
    # EJ5 secante
    # --------------------------
    def _build_ex5(self, tab: MethodTab) -> None:
        tab.x0 = self._add_entry(tab.input_frame, "x0:", "0.5", 0)
        tab.x1 = self._add_entry(tab.input_frame, "x1:", "1.0", 1)
        tab.tol = self._add_entry(tab.input_frame, "Tolerancia:", "1e-9", 2)
        tab.max_iter = self._add_entry(tab.input_frame, "Max iter:", "200", 3)

        ttk.Label(
            tab.input_frame,
            text="P(x)=x e^(-x/2) - 0.3\nSecante: no requiere derivada.",
            wraplength=260,
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=6, pady=(10, 4))

        tab.btn_calc.configure(command=lambda: self._calculate_ex5(tab))

    def _calculate_ex5(self, tab: MethodTab) -> None:
        tab.clear_all()

        try:
            x0 = float(tab.x0.get())
            x1 = float(tab.x1.get())
            tol = float(tab.tol.get())
            max_iter = int(float(tab.max_iter.get()))
        except ValueError as exc:
            messagebox.showerror("Error", f"Entradas inválidas: {exc}")
            return

        if not validate_tolerance(tol).ok or not validate_max_iterations(max_iter).ok:
            messagebox.showerror("Error", "Tolerancia o max iter inválidos.")
            return

        f = self.exercises["EX5"].f
        res = secant(f=f, x0=x0, x1=x1, tol=tol, max_iter=max_iter)

        for r in res.rows:
            tab.tree.insert("", "end", values=(
                r.n,
                format_number(r.x_prev),
                format_number(r.x_n),
                format_number(r.f_prev),
                format_number(r.f_n),
                format_number(r.x_next),
                format_number(r.abs_error),
            ))

        tab.final_vars.root.set(format_number(res.root))
        tab.final_vars.iterations.set(str(res.iterations))
        tab.final_vars.final_error.set(format_number(res.final_abs_error))
        tab.final_vars.elapsed.set(format_number(res.elapsed_seconds))
        tab.final_vars.status.set(res.message + f" | Eval f: {res.func_evals}")

        # Plot función + secantes
        tab.ax_func.clear()
        x_min, x_max = 0.0, 5.0
        xs = np.linspace(x_min, x_max, 600)
        ys = [f(float(x)) for x in xs]
        tab.ax_func.plot(xs, ys, label="P(x)")
        tab.ax_func.axhline(0.0)

        for ln in res.secant_lines[:10]:
            tab.ax_func.plot([ln.x1, ln.x2], [ln.y1, ln.y2], linestyle="--", alpha=0.7)

        tab.ax_func.set_title("Secante: función y secantes")
        tab.ax_func.legend()
        tab.canvas_func.draw()

        # Error log
        tab.ax_err.clear()
        tab.ax_err.set_title("Error absoluto (log) - Secante")
        tab.ax_err.set_xlabel("Iteración")
        tab.ax_err.set_ylabel("Error abs")
        if res.abs_errors:
            it = list(range(1, 1 + len(res.abs_errors)))
            tab.ax_err.semilogy(it, res.abs_errors, marker="o")
        tab.canvas_err.draw()

    # --------------------------
    # EJ5 comparación secante vs newton
    # --------------------------
    def _build_ex5_compare(self, tab: MethodTab) -> None:
        tab.tol = self._add_entry(tab.input_frame, "Tolerancia:", "1e-9", 0)
        tab.max_iter = self._add_entry(tab.input_frame, "Max iter:", "200", 1)
        tab.sec_x0 = self._add_entry(tab.input_frame, "Secante x0:", "0.5", 2)
        tab.sec_x1 = self._add_entry(tab.input_frame, "Secante x1:", "1.0", 3)
        tab.new_x0 = self._add_entry(tab.input_frame, "Newton x0:", "1.0", 4)

        ttk.Label(
            tab.input_frame,
            text="Comparación: Secante vs Newton (EX5)\nMétricas: iter, eval f, eval f', tiempo.",
            wraplength=260,
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=6, pady=(10, 4))

        tab.btn_calc.configure(command=lambda: self._calculate_ex5_compare(tab))

    def _calculate_ex5_compare(self, tab: MethodTab) -> None:
        tab.clear_all()

        try:
            tol = float(tab.tol.get())
            max_iter = int(float(tab.max_iter.get()))
            sec_x0 = float(tab.sec_x0.get())
            sec_x1 = float(tab.sec_x1.get())
            new_x0 = float(tab.new_x0.get())
        except ValueError as exc:
            messagebox.showerror("Error", f"Entradas inválidas: {exc}")
            return

        if not validate_tolerance(tol).ok or not validate_max_iterations(max_iter).ok:
            messagebox.showerror("Error", "Tolerancia o max iter inválidos.")
            return

        ex = self.exercises["EX5"]
        f = ex.f
        if ex.f_prime is None:
            messagebox.showerror("Error", "No existe derivada en EX5 (pero debería).")
            return

        fp = ex.f_prime

        res_sec = secant(f=f, x0=sec_x0, x1=sec_x1, tol=tol, max_iter=max_iter)
        res_new = newton(f=f, f_prime=fp, x0=new_x0, tol=tol, max_iter=max_iter)

        # tabla final (resumen)
        tab.tree.insert("", "end", values=("Secante", res_sec.iterations, res_sec.func_evals, 0, format_number(res_sec.elapsed_seconds)))
        tab.tree.insert("", "end", values=("Newton", res_new.iterations, res_new.func_evals, res_new.deriv_evals, format_number(res_new.elapsed_seconds)))

        tab.final_vars.root.set(f"Sec: {format_number(res_sec.root)} | New: {format_number(res_new.root)}")
        tab.final_vars.iterations.set(f"Sec: {res_sec.iterations} | New: {res_new.iterations}")
        tab.final_vars.final_error.set(f"Sec: {format_number(res_sec.final_abs_error)} | New: {format_number(res_new.final_abs_error)}")
        tab.final_vars.elapsed.set(f"Sec: {format_number(res_sec.elapsed_seconds)} | New: {format_number(res_new.elapsed_seconds)}")
        tab.final_vars.status.set("Comparación lista. Ver tabla de métricas.")

        # Plots: error log superpuesto
        tab.ax_func.clear()
        tab.ax_func.set_title("P(x) - referencia")
        xs = np.linspace(0.0, 5.0, 600)
        ys = [f(float(x)) for x in xs]
        tab.ax_func.plot(xs, ys, label="P(x)")
        tab.ax_func.axhline(0.0)
        tab.ax_func.legend()
        tab.canvas_func.draw()

        tab.ax_err.clear()
        tab.ax_err.set_title("Error abs (log) - Secante vs Newton")
        tab.ax_err.set_xlabel("Iteración")
        tab.ax_err.set_ylabel("Error abs")
        if res_sec.abs_errors:
            it_s = list(range(1, 1 + len(res_sec.abs_errors)))
            tab.ax_err.semilogy(it_s, res_sec.abs_errors, marker="o", label="Secante")
        if res_new.abs_errors:
            it_n = list(range(1, 1 + len(res_new.abs_errors)))
            tab.ax_err.semilogy(it_n, res_new.abs_errors, marker="x", label="Newton")
        tab.ax_err.legend()
        tab.canvas_err.draw()


def create_app() -> NumericMethodsApp:
    return NumericMethodsApp()