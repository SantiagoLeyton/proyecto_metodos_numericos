from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import math


Function = Callable[[float], float]


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    message: str


def validate_tolerance(tol: float) -> ValidationResult:
    if tol <= 0.0:
        return ValidationResult(False, "La tolerancia debe ser mayor que 0.")
    return ValidationResult(True, "OK")


def validate_max_iterations(max_iter: int) -> ValidationResult:
    if max_iter <= 0:
        return ValidationResult(False, "El máximo de iteraciones debe ser mayor que 0.")
    return ValidationResult(True, "OK")


def validate_bracket_sign_change(f: Function, a: float, b: float) -> ValidationResult:
    """Valida f(a)*f(b) < 0 (cambio de signo)."""
    fa = f(a)
    fb = f(b)

    if not math.isfinite(fa) or not math.isfinite(fb):
        return ValidationResult(False, "f(a) o f(b) no es finito (NaN/inf).")

    if fa == 0.0:
        return ValidationResult(True, "a es raíz exacta.")
    if fb == 0.0:
        return ValidationResult(True, "b es raíz exacta.")

    if fa * fb > 0.0:
        return ValidationResult(False, "El intervalo no contiene cambio de signo: f(a)*f(b) > 0.")
    return ValidationResult(True, "OK")


def safe_divide(numerator: float, denominator: float, eps: float = 1e-15) -> float:
    if abs(denominator) < eps:
        raise ZeroDivisionError("División por cero (o denominador muy pequeño).")
    return numerator / denominator


def absolute_error(current_x: float, previous_x: Optional[float]) -> Optional[float]:
    if previous_x is None:
        return None
    return abs(current_x - previous_x)


def relative_error_percent(current_x: float, previous_x: Optional[float]) -> Optional[float]:
    if previous_x is None:
        return None
    if current_x == 0.0:
        return None
    return abs(current_x - previous_x) / abs(current_x) * 100.0


def format_number(value: Optional[float], decimals: int = 8) -> str:
    """8 decimales mínimo + científico si es muy pequeño/grande."""
    if value is None:
        return "-"
    if not math.isfinite(value):
        return "NaN/Inf"
    abs_val = abs(value)
    if abs_val != 0.0 and (abs_val < 1e-6 or abs_val >= 1e6):
        return f"{value:.{decimals}e}"
    return f"{value:.{decimals}f}"


def detect_divergence(x: float, limit: float = 1e6) -> ValidationResult:
    if not math.isfinite(x) or abs(x) > limit:
        return ValidationResult(False, "Divergencia: |x_n| excede límite o no es finito.")
    return ValidationResult(True, "OK")