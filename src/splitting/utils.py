from functools import wraps
from typing import TypeVar, ParamSpec, Callable
import math

P = ParamSpec("P")
R = TypeVar("R")


def validate_splits(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        frac_train = kwargs.get("frac_train")
        frac_val = kwargs.get("frac_val")

        if not isinstance(frac_train, float):
            raise TypeError(
                f"frac_train must be float, got {type(frac_train).__name__}"
            )
        if not isinstance(frac_val, float):
            raise TypeError(f"frac_val must be float, got {type(frac_val).__name__}")

        frac_test = 1.0 - frac_train - frac_val

        if frac_test < 0.0 and not math.isclose(frac_test, 0.0, abs_tol=1e-12):
            raise ValueError(
                f"Invalid fractions: frac_train={frac_train}, "
                f"frac_val={frac_val}, frac_test={frac_test}"
            )

        if not math.isclose(frac_train + frac_val + frac_test, 1.0, abs_tol=1e-12):
            raise ValueError(
                f"Fractions must sum to 1, got "
                f"train={frac_train}, val={frac_val}, test={frac_test}"
            )
        return func(*args, **kwargs)

    return wrapper
