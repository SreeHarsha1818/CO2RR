"""Executable helpers for recovery caps and CO-dissociation step topologies."""

from typing import Dict, Tuple

MAX_RECOVERY_ATTEMPTS = 10
MAX_RECOVERY_ITERS = 10

def bounded_recovery_iters(max_recovery_iters: int = MAX_RECOVERY_ITERS,
                           max_recovery_attempts: int = MAX_RECOVERY_ATTEMPTS) -> int:
    return min(int(max_recovery_iters), int(max_recovery_attempts))

def get_recovery_caps(cfg: Dict) -> Tuple[int, int]:
    max_attempts = int(cfg.get("max_recovery_attempts", MAX_RECOVERY_ATTEMPTS))
    cfg_iters = int(cfg.get("max_recovery_iters", max_attempts))
    return max_attempts, min(cfg_iters, max_attempts)

CO_DISSOCIATION_TOPOLOGY_RULES = {
    "05_OH_from_CO_diss": {
        "surf_atom": "any",
        "req_bonds": [("O", "H")],
        "forb_bonds": [("C", "O"), ("C", "H")],
        "co_min": None, "co_max": None, "co_triple": False,
        "oh_count": 1, "ch_count": 0,
    },
    "05_H2O_from_CO_diss": {
        "surf_atom": "any",
        "req_bonds": [("O", "H")],
        "forb_bonds": [("C", "O"), ("C", "H")],
        "co_min": None, "co_max": None, "co_triple": False,
        "oh_count": 2, "ch_count": 0, "h2o_present": True,
    },
    "06_C_from_CO_diss": {
        "surf_atom": "C",
        "req_bonds": [],
        "forb_bonds": [("C", "O"), ("C", "H"), ("O", "H")],
        "co_min": None, "co_max": None, "co_triple": False,
        "oh_count": 0, "ch_count": 0,
    },
}
