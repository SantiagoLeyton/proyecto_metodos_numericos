from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import time

from utils.validaciones import absolute_error, relative_error_percent, safe_divide, detect_divergence


Function = Callable[[float], float]


@dataclass(frozen=True)
class NewtonIteration:
    n: int
    x_n: float
    fx: float
    fpx: float
    abs_error: Optional[float]
    rel_error_pct: Optional[float]


@dataclass(frozen=True)
class NewtonTangent:
    """Representa la recta tangente en x_n: y = f(x_n) + f'(x_n)*(x - x_n)."""
    x0: float
    y0: float
    slope: float


@dataclass(frozen=True)
class NewtonResult:
    converged: bool
    root: float
    iterations: int
    final_abs_error: Optional[float]
    elapsed_seconds: float
    message: str
    rows: List[NewtonIteration]
    x_values: List[float]
    abs_errors: List[float]
    tangents: List[NewtonTangent]
    func_evals: int
    deriv_evals: int


def newton(
    f: Function,
    f_prime: Function,
    x0: float,
    tol: float,
    max_iter: int,
    divergence_limit: float = 1e6,
) -> NewtonResult:
    start = time.perf_counter()

    func_evals = 0
    deriv_evals = 0

    rows: List[NewtonIteration] = []
    x_values: List[float] = [x0]
    abs_errors_series: List[float] = []
    tangents: List[NewtonTangent] = []

    prev_x: Optional[float] = None
    x_n = x0

    for n in range(1, max_iter + 1):
        fx = f(x_n); func_evals += 1
        fpx = f_prime(x_n); deriv_evals += 1

        if fpx == 0.0:
            elapsed = time.perf_counter() - start
            return NewtonResult(False, x_n, n - 1, None, elapsed,
                               "Falla: f'(x_n)=0, no se puede dividir.",
                               rows, x_values, abs_errors_series, tangents, func_evals, deriv_evals)

        tangents.append(NewtonTangent(x0=x_n, y0=fx, slope=fpx))

        step = safe_divide(fx, fpx)
        x_next = x_n - step

        abs_err = absolute_error(x_next, x_n)
        rel_err = relative_error_percent(x_next, x_n)

        rows.append(NewtonIteration(n, x_n, fx, fpx, abs_err, rel_err))
        x_values.append(x_next)
        if abs_err is not None:
            abs_errors_series.append(abs_err)

        dv = detect_divergence(x_next, limit=divergence_limit)
        if not dv.ok:
            elapsed = time.perf_counter() - start
            return NewtonResult(False, x_next, n, abs_err, elapsed,
                               f"Falla: {dv.message}",
                               rows, x_values, abs_errors_series, tangents, func_evals, deriv_evals)

        if abs_err is not None and abs_err <= tol:
            elapsed = time.perf_counter() - start
            return NewtonResult(True, x_next, n, abs_err, elapsed,
                               "Convergencia exitosa: tolerancia alcanzada.",
                               rows, x_values, abs_errors_series, tangents, func_evals, deriv_evals)

        x_n = x_next
        prev_x = x_n

    elapsed = time.perf_counter() - start
    final_abs = rows[-1].abs_error if rows else None
    return NewtonResult(False, x_n, len(rows), final_abs, elapsed,
                       "No convergió: se alcanzó el máximo de iteraciones.",
                       rows, x_values, abs_errors_series, tangents, func_evals, deriv_evals)