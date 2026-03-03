from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import time
import math

from utils.validaciones import absolute_error, relative_error_percent, safe_divide


Function = Callable[[float], float]


@dataclass(frozen=True)
class BisectionIteration:
    """Una fila de iteración para el método de bisección."""
    n: int
    a: float
    b: float
    c: float
    fc: float
    abs_error: Optional[float]
    rel_error_pct: Optional[float]


@dataclass(frozen=True)
class BisectionResult:
    """Resultado completo del método de bisección."""
    converged: bool
    root: float
    iterations: int
    final_abs_error: Optional[float]
    elapsed_seconds: float
    message: str
    rows: List[BisectionIteration]
    c_values: List[float]
    abs_errors: List[float]


def bisection(
    f: Function,
    a: float,
    b: float,
    tol: float,
    max_iter: int,
) -> BisectionResult:
    """
    Implementa el método de bisección para encontrar una raíz en [a, b].

    Criterio de parada:
    - abs_error = |c_n - c_{n-1}| <= tol
    - o max_iter alcanzado
    """
    start = time.perf_counter()

    fa = f(a)
    fb = f(b)

    if fa == 0.0:
        elapsed = time.perf_counter() - start
        return BisectionResult(
            converged=True,
            root=a,
            iterations=0,
            final_abs_error=0.0,
            elapsed_seconds=elapsed,
            message="Convergencia: a es raíz exacta.",
            rows=[],
            c_values=[a],
            abs_errors=[],
        )

    if fb == 0.0:
        elapsed = time.perf_counter() - start
        return BisectionResult(
            converged=True,
            root=b,
            iterations=0,
            final_abs_error=0.0,
            elapsed_seconds=elapsed,
            message="Convergencia: b es raíz exacta.",
            rows=[],
            c_values=[b],
            abs_errors=[],
        )

    if fa * fb > 0.0:
        elapsed = time.perf_counter() - start
        return BisectionResult(
            converged=False,
            root=float("nan"),
            iterations=0,
            final_abs_error=None,
            elapsed_seconds=elapsed,
            message="Falla: el intervalo no tiene cambio de signo (f(a)*f(b) > 0).",
            rows=[],
            c_values=[],
            abs_errors=[],
        )

    rows: List[BisectionIteration] = []
    c_values: List[float] = []
    abs_errors_series: List[float] = []

    prev_c: Optional[float] = None
    final_abs_err: Optional[float] = None

    for n in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c)

        abs_err = absolute_error(c, prev_c)
        rel_err = relative_error_percent(c, prev_c)

        rows.append(
            BisectionIteration(
                n=n,
                a=a,
                b=b,
                c=c,
                fc=fc,
                abs_error=abs_err,
                rel_error_pct=rel_err,
            )
        )
        c_values.append(c)
        if abs_err is not None:
            abs_errors_series.append(abs_err)

        # Criterio de parada por tolerancia
        if abs_err is not None and abs_err <= tol:
            final_abs_err = abs_err
            elapsed = time.perf_counter() - start
            return BisectionResult(
                converged=True,
                root=c,
                iterations=n,
                final_abs_error=final_abs_err,
                elapsed_seconds=elapsed,
                message="Convergencia exitosa: tolerancia alcanzada.",
                rows=rows,
                c_values=c_values,
                abs_errors=abs_errors_series,
            )

        # Si fc es exactamente cero, terminamos
        if fc == 0.0:
            final_abs_err = abs_err if abs_err is not None else 0.0
            elapsed = time.perf_counter() - start
            return BisectionResult(
                converged=True,
                root=c,
                iterations=n,
                final_abs_error=final_abs_err,
                elapsed_seconds=elapsed,
                message="Convergencia: f(c)=0 (raíz exacta).",
                rows=rows,
                c_values=c_values,
                abs_errors=abs_errors_series,
            )

        # Actualización del intervalo
        if fa * fc < 0.0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        prev_c = c

    elapsed = time.perf_counter() - start
    final_abs_err = rows[-1].abs_error if rows else None
    return BisectionResult(
        converged=False,
        root=c_values[-1] if c_values else float("nan"),
        iterations=len(rows),
        final_abs_error=final_abs_err,
        elapsed_seconds=elapsed,
        message="No convergió: se alcanzó el máximo de iteraciones.",
        rows=rows,
        c_values=c_values,
        abs_errors=abs_errors_series,
    )