from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional
import time

from utils.validaciones import absolute_error, relative_error_percent, safe_divide


Function = Callable[[float], float]


@dataclass(frozen=True)
class FalsePositionIteration:
    n: int
    a: float
    b: float
    c: float
    fc: float
    abs_error: Optional[float]
    rel_error_pct: Optional[float]


@dataclass(frozen=True)
class FalsePositionResult:
    converged: bool
    root: float
    iterations: int
    final_abs_error: Optional[float]
    elapsed_seconds: float
    message: str
    rows: List[FalsePositionIteration]
    c_values: List[float]
    abs_errors: List[float]
    func_evals: int


def false_position(f: Function, a: float, b: float, tol: float, max_iter: int) -> FalsePositionResult:
    start = time.perf_counter()

    func_evals = 0
    fa = f(a); func_evals += 1
    fb = f(b); func_evals += 1

    if fa == 0.0:
        elapsed = time.perf_counter() - start
        return FalsePositionResult(True, a, 0, 0.0, elapsed, "Convergencia: a es raíz exacta.",
                                  [], [a], [], func_evals)
    if fb == 0.0:
        elapsed = time.perf_counter() - start
        return FalsePositionResult(True, b, 0, 0.0, elapsed, "Convergencia: b es raíz exacta.",
                                  [], [b], [], func_evals)

    if fa * fb > 0.0:
        elapsed = time.perf_counter() - start
        return FalsePositionResult(False, float("nan"), 0, None, elapsed,
                                  "Falla: el intervalo no tiene cambio de signo (f(a)*f(b) > 0).",
                                  [], [], [], func_evals)

    rows: List[FalsePositionIteration] = []
    c_values: List[float] = []
    abs_errors_series: List[float] = []

    prev_c: Optional[float] = None

    for n in range(1, max_iter + 1):
        denom = (fb - fa)
        try:
            frac = safe_divide(fb * (b - a), denom)
        except ZeroDivisionError:
            elapsed = time.perf_counter() - start
            return FalsePositionResult(False, float("nan"), len(rows), None, elapsed,
                                      "Falla: división por cero en (f(b)-f(a)).",
                                      rows, c_values, abs_errors_series, func_evals)

        c = b - frac
        fc = f(c); func_evals += 1

        abs_err = absolute_error(c, prev_c)
        rel_err = relative_error_percent(c, prev_c)

        rows.append(FalsePositionIteration(n, a, b, c, fc, abs_err, rel_err))
        c_values.append(c)
        if abs_err is not None:
            abs_errors_series.append(abs_err)

        if abs_err is not None and abs_err <= tol:
            elapsed = time.perf_counter() - start
            return FalsePositionResult(True, c, n, abs_err, elapsed,
                                      "Convergencia exitosa: tolerancia alcanzada.",
                                      rows, c_values, abs_errors_series, func_evals)

        if fc == 0.0:
            elapsed = time.perf_counter() - start
            return FalsePositionResult(True, c, n, abs_err or 0.0, elapsed,
                                      "Convergencia: f(c)=0 (raíz exacta).",
                                      rows, c_values, abs_errors_series, func_evals)

        # Actualización de intervalo
        if fa * fc < 0.0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        prev_c = c

    elapsed = time.perf_counter() - start
    final_abs = rows[-1].abs_error if rows else None
    return FalsePositionResult(False, c_values[-1], len(rows), final_abs, elapsed,
                              "No convergió: se alcanzó el máximo de iteraciones.",
                              rows, c_values, abs_errors_series, func_evals)