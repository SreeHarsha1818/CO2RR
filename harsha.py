"""
Helper module with the requested updated recovery/topology settings.

This mirrors the current recovery cap semantics used in run_mace_phonons.py
so the key values are easy to find in one place.
"""

# Recovery hard caps
MAX_RECOVERY_ATTEMPTS = 10
MAX_RECOVERY_ITERS = 10


def bounded_recovery_iters(max_recovery_iters: int = MAX_RECOVERY_ITERS,
                           max_recovery_attempts: int = MAX_RECOVERY_ATTEMPTS) -> int:
    """Return hard-bounded recovery iterations (same policy as workflow)."""
    return min(int(max_recovery_iters), int(max_recovery_attempts))


# Added CO-dissociation topology keys
CO_DISSOCIATION_TOPOLOGY_KEYS = (
    "05_OH_from_CO_diss",
    "05_H2O_from_CO_diss",
    "06_C_from_CO_diss",
)


UPDATED_NOTE = (
    "Recovery loop is bounded; escalating random per-cycle perturbation "
    "is removed; CO-dissociation topologies are explicitly defined."
)
