from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import time

from utils.validaciones import absolute_error, safe_divide, detect_divergence


Function = Callable[[float], float]


@dataclass(frozen=True)
class SecantIteration:
    n: int
    x_prev: float
    x_n: float
    f_prev: float
    f_n: float
    x_next: float
    abs_error: Optional[float]


@dataclass(frozen=True)
class SecantLine:
    """Segmento secante entre (x_prev, f_prev) y (x_n, f_n)."""
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class SecantResult:
    converged: bool
    root: float
    iterations: int
    final_abs_error: Optional[float]
    elapsed_seconds: float
    message: str
    rows: List[SecantIteration]
    x_values: List[float]
    abs_errors: List[float]
    secant_lines: List[SecantLine]
    func_evals: int


def secant(
    f: Function,
    x0: float,
    x1: float,
    tol: float,
    max_iter: int,
    divergence_limit: float = 1e6,
) -> SecantResult:
    start = time.perf_counter()

    func_evals = 0
    f0 = f(x0); func_evals += 1
    f1 = f(x1); func_evals += 1

    rows: List[SecantIteration] = []
    x_values: List[float] = [x0, x1]
    abs_errors_series: List[float] = []
    lines: List[SecantLine] = []

    for n in range(1, max_iter + 1):
        denom = (f1 - f0)
        if denom == 0.0:
            elapsed = time.perf_counter() - start
            return SecantResult(False, x1, n - 1, None, elapsed,
                               "Falla: f(x_n)-f(x_{n-1})=0 (división por cero).",
                               rows, x_values, abs_errors_series, lines, func_evals)

        lines.append(SecantLine(x1=x0, y1=f0, x2=x1, y2=f1))

        x2 = x1 - safe_divide(f1 * (x1 - x0), denom)
        err = absolute_error(x2, x1)

        rows.append(SecantIteration(n, x0, x1, f0, f1, x2, err))
        x_values.append(x2)
        if err is not None:
            abs_errors_series.append(err)

        dv = detect_divergence(x2, limit=divergence_limit)
        if not dv.ok:
            elapsed = time.perf_counter() - start
            return SecantResult(False, x2, n, err, elapsed,
                               f"Falla: {dv.message}",
                               rows, x_values, abs_errors_series, lines, func_evals)

        if err is not None and err <= tol:
            elapsed = time.perf_counter() - start
            return SecantResult(True, x2, n, err, elapsed,
                               "Convergencia exitosa: tolerancia alcanzada.",
                               rows, x_values, abs_errors_series, lines, func_evals)

        # avanzar
        x0, x1 = x1, x2
        f0, f1 = f1, f(x1); func_evals += 1

    elapsed = time.perf_counter() - start
    final_abs = rows[-1].abs_error if rows else None
    return SecantResult(False, x1, len(rows), final_abs, elapsed,
                       "No convergió: se alcanzó el máximo de iteraciones.",
                       rows, x_values, abs_errors_series, lines, func_evals)