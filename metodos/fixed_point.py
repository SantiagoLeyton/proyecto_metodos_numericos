from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import time

from utils.validaciones import absolute_error, relative_error_percent, detect_divergence


Function = Callable[[float], float]


@dataclass(frozen=True)
class FixedPointIteration:
    n: int
    x_n: float
    g_xn: float
    residual: float
    rel_error_pct: Optional[float]
    abs_error: Optional[float]


@dataclass(frozen=True)
class FixedPointResult:
    converged: bool
    root: float
    iterations: int
    final_abs_error: Optional[float]
    elapsed_seconds: float
    message: str
    rows: List[FixedPointIteration]
    x_values: List[float]
    abs_errors: List[float]
    gprime_at_start: Optional[float]
    func_evals: int


def fixed_point(
    g: Function,
    x0: float,
    tol: float,
    max_iter: int,
    g_prime: Optional[Function] = None,
    divergence_limit: float = 1e6,
) -> FixedPointResult:
    start = time.perf_counter()

    func_evals = 0

    gprime0: Optional[float] = None
    if g_prime is not None:
        try:
            gprime0 = g_prime(x0)
        except Exception:
            gprime0 = None

    rows: List[FixedPointIteration] = []
    x_values: List[float] = [x0]
    abs_errors_series: List[float] = []

    prev_x: Optional[float] = None
    x_n = x0

    for n in range(1, max_iter + 1):
        g_xn = g(x_n); func_evals += 1
        x_next = g_xn

        abs_err = absolute_error(x_next, x_n)
        rel_err = relative_error_percent(x_next, x_n)
        residual = abs(x_n - g_xn)

        rows.append(FixedPointIteration(n, x_n, g_xn, residual, rel_err, abs_err))
        x_values.append(x_next)
        if abs_err is not None:
            abs_errors_series.append(abs_err)

        # Divergencia
        dv = detect_divergence(x_next, limit=divergence_limit)
        if not dv.ok:
            elapsed = time.perf_counter() - start
            return FixedPointResult(
                converged=False,
                root=x_next,
                iterations=n,
                final_abs_error=abs_err,
                elapsed_seconds=elapsed,
                message=f"Falla: {dv.message}",
                rows=rows,
                x_values=x_values,
                abs_errors=abs_errors_series,
                gprime_at_start=gprime0,
                func_evals=func_evals,
            )

        # Criterio de parada: |x_{n+1} - x_n| <= tol
        if abs_err is not None and abs_err <= tol:
            elapsed = time.perf_counter() - start
            return FixedPointResult(
                converged=True,
                root=x_next,
                iterations=n,
                final_abs_error=abs_err,
                elapsed_seconds=elapsed,
                message="Convergencia exitosa: tolerancia alcanzada.",
                rows=rows,
                x_values=x_values,
                abs_errors=abs_errors_series,
                gprime_at_start=gprime0,
                func_evals=func_evals,
            )

        prev_x = x_n
        x_n = x_next

    elapsed = time.perf_counter() - start
    final_abs = rows[-1].abs_error if rows else None
    return FixedPointResult(
        converged=False,
        root=x_n,
        iterations=len(rows),
        final_abs_error=final_abs,
        elapsed_seconds=elapsed,
        message="No convergió: se alcanzó el máximo de iteraciones.",
        rows=rows,
        x_values=x_values,
        abs_errors=abs_errors_series,
        gprime_at_start=gprime0,
        func_evals=func_evals,
    )