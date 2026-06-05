# Legacy compatibility wrapper.
# Use pid_optimizer.py for new code.

from pid_optimizer import PIDOptimizer, PIDResult

__all__ = [
    "PIDResult",
    "PIDOptimizer",
]
