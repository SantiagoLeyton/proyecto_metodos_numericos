from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import math


Function = Callable[[float], float]


@dataclass(frozen=True)
class ExerciseDefinition:
    """Definición de una función (y opcionalmente derivadas) para un ejercicio."""
    key: str
    display_name: str
    f: Function
    f_prime: Optional[Function] = None
    g: Optional[Function] = None
    g_prime: Optional[Function] = None
    domain_hint: Optional[Tuple[float, float]] = None


# =========================
# EJERCICIO 1
# T(λ) = 2.5 + 0.8λ² - 3.2λ + ln(λ + 1)
# =========================
def exercise1_f(x: float) -> float:
    """Función T(λ). Dominio: x > -1 por ln(x+1)."""
    if x <= -1.0:
        raise ValueError("Dominio inválido: se requiere x > -1 por ln(x+1).")
    return 2.5 + 0.8 * (x ** 2) - 3.2 * x + math.log(x + 1.0)


# =========================
# EJERCICIO 2
# E(x) = x³ - 6x² + 11x - 6.5
# =========================
def exercise2_f(x: float) -> float:
    return (x ** 3) - 6.0 * (x ** 2) + 11.0 * x - 6.5


# =========================
# EJERCICIO 3 (Punto fijo)
# x = 0.5cos(x) + 1.5  => g(x) = 0.5cos(x) + 1.5
# =========================
def exercise3_g(x: float) -> float:
    return 0.5 * math.cos(x) + 1.5


def exercise3_g_prime(x: float) -> float:
    """g'(x) = -0.5 sin(x)"""
    return -0.5 * math.sin(x)


def exercise3_fixed_point_residual(x: float) -> float:
    """Para graficar como raíz: f(x)=x-g(x)."""
    return x - exercise3_g(x)


# =========================
# EJERCICIO 4 (Newton)
# T(n) = n³ - 8n² + 20n - 16
# =========================
def exercise4_f(x: float) -> float:
    return (x ** 3) - 8.0 * (x ** 2) + 20.0 * x - 16.0


def exercise4_f_prime(x: float) -> float:
    """T'(n) = 3n² - 16n + 20"""
    return 3.0 * (x ** 2) - 16.0 * x + 20.0


# =========================
# EJERCICIO 5 (Secante vs Newton)
# P(x) = x e^(-x/2) - 0.3
# f'(x)= e^(-x/2)(1 - x/2)
# =========================
def exercise5_f(x: float) -> float:
    return x * math.exp(-x / 2.0) - 0.3


def exercise5_f_prime(x: float) -> float:
    return math.exp(-x / 2.0) * (1.0 - x / 2.0)


def get_exercises() -> Dict[str, ExerciseDefinition]:
    exercises: Dict[str, ExerciseDefinition] = {
        "EX1": ExerciseDefinition(
            key="EX1",
            display_name="Ejercicio 1 - Bisección (Hash Table)",
            f=exercise1_f,
            domain_hint=(0.5, 2.5),
        ),
        "EX2": ExerciseDefinition(
            key="EX2",
            display_name="Ejercicio 2 - Comparación (Balanceo)",
            f=exercise2_f,
            domain_hint=(2.0, 4.0),
        ),
        "EX3": ExerciseDefinition(
            key="EX3",
            display_name="Ejercicio 3 - Punto Fijo (Crecimiento BD)",
            f=exercise3_fixed_point_residual,
            g=exercise3_g,
            g_prime=exercise3_g_prime,
            domain_hint=(0.0, 3.0),
        ),
        "EX4": ExerciseDefinition(
            key="EX4",
            display_name="Ejercicio 4 - Newton-Raphson (Concurrencia)",
            f=exercise4_f,
            f_prime=exercise4_f_prime,
            domain_hint=(0.0, 6.0),
        ),
        "EX5": ExerciseDefinition(
            key="EX5",
            display_name="Ejercicio 5 - Secante vs Newton (Escalabilidad)",
            f=exercise5_f,
            f_prime=exercise5_f_prime,
            domain_hint=(0.0, 5.0),
        ),
    }
    return exercises