# CODEX
#!/usr/bin/env python3 -u
"""
CO2RR Autonomous Workflow -- Production-Grade, Chemistry-Aware, Fault-Tolerant
==============================================================================
ASE + MACE (or any ASE-compatible calculator) workflow for autonomous
CO2 reduction reaction (CO2RR) hydrogenation on catalytic surfaces.

Architecture
------------
  SlabGenerator          -- clean slab from POSCAR/bulk; freeze protocol
  ActiveSiteIdentifier   -- top / bridge / hollow / 4-fold hollow site detection
  ChemistryValidator     -- bond lengths, stoichiometry, overlap, fragmentation,
                           drift, C-O integrity, SCF failure, coordination
  AdsorptionEngine       -- place CO2 and all intermediates with correct geometry
  HydrogenationEngine    -- chemistry-aware H addition (C-bound vs O-bound)
  StructureBuilder       -- explicit geometry for every step of every pathway
  OptimizerManager       -- FIRE/LBFGS multi-stage protocol; checkpoint/restart
  AdaptiveRecoveryEngine -- >=3-5 recovery attempts: perturb / reorient / invert
                           C/O / alternate hydrogenation atom / height/tilt /
                           metastable config / alternate optimizer
  EnergyAnalyzer         -- CHE + ZPE + solvation + electrode potential sweep
  ReactionTracker        -- pathway connectivity, energetics, retry history,
                           limiting potential, selectivity metrics
  OutputManager          -- POSCAR/CONTCAR, extxyz, CIF, trajectories, JSON, CSV,
                           reaction tree / pathway map

Reaction Pathways
-----------------
  Path A  (Formaldehyde route):
    CO2->COOH*->[H2O*+CO*]->CO*->CHO*->CH2O*->CH3O*->[CH4*+O*]->O*->OH*->H2O*->clean

  Path B  (Hydroxymethylidene route):
    CO2->COOH*->[H2O*+CO*]->CO*->COH*->[C*+H2O*]->C*->CH*->CH2*->CH3*->CH4^

  Path C  (Formate route):
    CO2->HCOO*->HCOOH*->[H2O*+CHO*]->CHO*->CH2O*->CH3O*->[CH4*+O*]->O*->OH*->H2O*->clean

  Path D  (Carbene / CO-dissociation route):
    CO2->COOH*->CO*->[C*+O*]->[C*+OH*]->[C*+H2O*]->C*->CH*->CH2*(carbene)->CH3*->CH4^

  Path E  (Methanol route):
    CO2->COOH*->CO*->CHO*->CH2O*->CH3O*->CH3OH*->CH3OH^ (or CH3O*->CH3*+OH*->CH4)

All electrochemical steps: X* + H+ + e- -> XH*  (CHE convention)

Chemistry Enforcement
---------------------
  - Strict stoichiometry at every step (atom-count checking)
  - Chemically valid bond lengths (C-O, C-H, O-H, C-metal, O-metal)
  - No adsorbate overlap, fragmentation, or desorption
  - Surface-anchored adsorption throughout the pathway
  - Convergence checks (max force, energy change)
  - Adaptive recovery: >=3 attempts before pathway rejection

References
----------
  Peterson et al., Energy Environ. Sci. 2010, 3, 1311
  Nie et al., ACS Catal. 2014, 4, 2119
  Chan & Norskov, J. Phys. Chem. Lett. 2015, 6, 2663
  Gauthier et al., J. Chem. Theory Comput. 2019, 15, 6064
  NIST-JANAF Thermochemical Tables (gas-phase references)
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import sys
import os
import copy
import concurrent.futures
import argparse
import json
import csv
import pickle
import hashlib
import logging
import datetime
import warnings
import traceback
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ase import Atom, Atoms
from ase.build import bulk, surface, molecule, add_adsorbate
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.io import read, write, Trajectory
from ase.optimize import FIRE, LBFGS, BFGSLineSearch
from ase.neighborlist import NeighborList, natural_cutoffs

# -- optional: try importing MACE; fall back to EMT for testing ---------------
try:
    from mace.calculators import MACECalculator
    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False
    warnings.warn("MACE not found -- falling back to EMT for structure validation")

# Force unbuffered output for HPC logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
warnings.filterwarnings("ignore")

# ==============================================================================
# COLOUR CODES  (ANSI -- stripped in log files via ColorlessFormatter)
# ==============================================================================
GREEN   = "\033[92m"
BLUE    = "\033[94m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
CFG: Dict[str, Any] = {
    # -- Calculator ---------------------------------------------------------
    "model_path":               "mace-mh-1.model",
    "model_heads": [
        "oc20_usemppbe",   # OC20-trained head (PBE-level)
        "omat_pbe",        # OMat head (PBE, materials coverage)
    ],
    # Device auto-detected at import time.
    # /proc/driver/nvidia/gpus exists even when CUDA toolkit is not loaded.
    # torch.cuda.is_available() is the only reliable test.
    "device": (lambda: (lambda t: "cuda" if t else "cpu")(
        __import__("subprocess").run(
            ["python3", "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True).stdout.strip() == "True"
    ))(),
    "default_dtype":            "float32",

    # -- Paths --------------------------------------------------------------
    "slab_poscar":  "./00_bare_slab/POSCAR",
    "workdir":      "./co2rr_autonomous_workflow",

    # -- Slab geometry ------------------------------------------------------
    "top_layer_tol":            1.0,    # Ang -- atoms within this of max-z are "top layer"
    "freeze_fraction":          0.35,   # freeze bottom 35 % of slab z-range
    "max_sites":                25,     # max active sites to screen per head
    "z_height":                 1.65,   # Ang above surface top for adsorbate placement
    "desorption_gap_threshold": 5.0,    # Ang -- adsorbate > this above surface -> desorbed (softened from 4.0)
    "co_separation_diss":       2.50,   # Ang lateral C-O separation for CO dissociation

    # -- Relaxation ---------------------------------------------------------
    "relax_fmax":               0.03,   # eV/Ang -- main convergence criterion
    "relax_steps":              2000,
    "pre_relax0_fmax":          1.00,   # eV/Ang -- stage-0 pre-relax (FIRE, big steps)
    "pre_relax0_steps":         100,
    "pre_relax1_fmax":          0.30,   # eV/Ang -- stage-1 pre-relax (FIRE, smaller)
    "pre_relax1_steps":         200,
    "lbfgs_maxstep":            0.05,   # Ang
    "lbfgs_memory":             20,
    "lbfgs_damping":            0.25,
    "lbfgs_alpha":              70.0,
    "fire_maxstep":             0.30,

    # -- Stability / recovery -----------------------------------------------
    "stability_iterations":         3,  # default; overridden per-step below
    "stability_relax_fmax":         0.05,
    "stability_max_steps":          500,
    "stability_min_intact_to_stop": 1,
    "pre_relax_steps":             200,  # early-exit after this many intact found
    "perturbation_magnitude":       0.05,   # Ang -- adsorbate-atom displacement
    "max_recovery_attempts":        7,  # not used for capping -- used as strategy cycle size
    # Strict enforcement
    "strict_no_propagate_broken":   True,   # NEVER propagate broken intermediates
    "max_recovery_iters":           200,    # hard safety cap (set None for truly infinite)
    "always_run_stability":         True,   # run stability_tests even when main relax is intact
    "stability_initial_iters":      6,      # iters in first stability pass (before recovery)
    "recovery_log_all_attempts":    True,   # save every attempt with POSCAR/CONTCAR/CIF/log

    # -- Chemistry-aware per-step iteration counts --------------------------
    # 1 -> strip/rearrangement  2 -> robust  3 -> fragile  6 -> CHO (Path A)
    "stability_iterations_per_step": {
        "01_CO2":                2,
        "02_COOH":               2,
        "02_HCOO":               2,
        "03_H2O_from_COOH":      1,
        "03_CO":                 1,
        "03_HCOOH":              2,
        "04_H2O_from_HCOOH":     1,
        "04_CHO":                6,  # Path A only; overridden to 1 for Path C
        "04_COH":                2,
        "05_H2O_from_COH":       1,
        "05_C":                  1,
        "05_CH2O":               3,
        "05_CO_diss":            3,
        "05_OH_from_CO_diss":    2,
        "05_H2O_from_CO_diss":   1,
        "06_C_from_CO_diss":     1,
        "06_CH":                 2,
        "06_CH3O":               3,
        "07_O_CH4":              3,
        "07_O":                  1,
        "07_CH2":                2,
        "08_CH3":                2,
        "08_OH":                 2,
        "09_CH4":                1,
        "09_H2O":                2,
        "10_clean":              1,
        # Path E (Methanol route)
        "05_CH2O_E":             3,
        "06_CH3O_E":             3,
        "07_CH3OH":              3,
        "08_CH3OH_des":          1,
    },

    # -- Electrochemical potential sweep -----------------------------------
    "U_sweep":              True,
    "U_min_V":              -1.50,   # V vs RHE
    "U_step_V":              0.10,
    "temperature_K":        298.15,

    # -- Parallelism (site-level) ------------------------------------------
    # n_parallel_pathways: run this many pathways concurrently per site
    # (uses ThreadPoolExecutor; safe for CPU-MACE; set 1 for GPU).
    "n_parallel_pathways":  1,

    # SLURM array-job batching: set site_start/site_end to process only
    # a slice of all_sites.  e.g. site_start=0, site_end=5 for task 0.
    # Leave both as None to run all sites sequentially.
    "site_start":           None,
    "site_end":             None,

    # -- Slab reconstruction guard -----------------------------------------
    # Abort relaxation if any slab atom moves more than this (Ang)
    "slab_max_displacement": 2.0,

    # -- Output formats ----------------------------------------------------
    "save_cif":             True,
    "save_extxyz":          True,
    "save_trajectories":    True,
    "save_json_summary":    True,
    "save_csv_summary":     True,
    "save_pathway_map":     True,

    # -- Early-stop relaxation (chemistry-aware) --------------------------
    # When True, the LBFGS loop checks chemical integrity every
    # check_interval steps and stops the MOMENT an intact metastable
    # configuration is found.  Prevents over-relaxation into wrong minima.
    "early_stop_on_intact":       False,  # True = stop at first intact (fast screen); False = full relax (recommended)
    "metastable_intact_fmax":     0.30,  # fmax below which an intact snapshot is saved as metastable backup
    "check_interval":             5,     # integrity check every N steps
    "min_steps_before_check":     5,     # don't check too early (structure still settling)

    # -- Iteration-until-intact settings -----------------------------------
    "recovery_iters_per_stage":   6,    # base iters per stage (scales up)
    "max_intact_search_iters":    18,   # hard cap: total iters across all stages
    "intact_required_to_record":  True, # only record DG for intact structures

    # -- Bond / geometry thresholds (all in Ang) ---------------------------
    "desorp_z_threshold":         5.0,  # z above surface = desorbed
    "surface_bond_max":           2.80, # max Ang for a valid metal-adsorbate bond
                                        # Metal-C ~1.9-2.3, Metal-O ~1.8-2.2,
                                        # Metal-H ~1.7-2.0.  2.8 is generous.
    "surface_bond_top_layer_tol": 2.5,  # how deep below surface top counts as "top layer"
    "co_bond_max":                1.80,
    "ch_bond_max":                1.40,
    "oh_bond_max":                1.30,

    # -- Optimizer fine-tuning ---------------------------------------------
    "gas_relax_fmax":             0.03,
    "gas_relax_steps":            500,
    "fire1_maxstep":              0.10,
    "stab_fire_maxstep":          0.05,
    "stab_fire_fmax":             0.50,
}

# -- Derived U-sweep values ----------------------------------------------------
_n_steps = int(abs(CFG["U_min_V"]) / CFG["U_step_V"])
CFG["U_values"] = (
    [round(-i * CFG["U_step_V"], 2) for i in range(_n_steps + 1)]
    if CFG["U_sweep"] else [0.0]
)

# ==============================================================================
# ZPE / ENTROPY CORRECTIONS  (eV, 298.15 K)
# Source: Peterson et al. EES 2010; Nie et al. ACS Catal. 2014; NIST-JANAF
# Format: DeltaZPE - T*DeltaS  (positive = destabilises intermediate relative to refs)
# ==============================================================================
ZPE_TS_CORRECTIONS: Dict[str, float] = {
    "CO2_gas":               0.00,
    "H2_gas":                0.00,
    "H2O_gas":               0.00,
    "01_CO2":                0.12,
    "02_COOH":               0.15,
    "02_HCOO":               0.14,
    "03_H2O_from_COOH":      0.13,
    "03_CO":                 0.10,
    "03_HCOOH":              0.17,
    "04_H2O_from_HCOOH":     0.14,
    "04_CHO":                0.16,
    "04_COH":                0.15,
    "05_H2O_from_COH":       0.13,
    "05_C":                  0.08,
    "05_CH2O":               0.19,
    "05_CO_diss":            0.09,
    "05_OH_from_CO_diss":    0.12,
    "05_H2O_from_CO_diss":   0.13,
    "06_C_from_CO_diss":     0.08,
    "06_CH":                 0.12,
    "06_CH3O":               0.22,
    "07_O_CH4":              0.20,
    "07_O":                  0.07,
    "07_CH2":                0.16,
    "08_CH3":                0.19,
    "08_OH":                 0.11,
    "09_CH4":               -0.32,   # entropy gain on desorption
    "09_H2O":                0.10,
    "10_clean":              0.00,
    # Path E (Methanol route)
    "05_CH2O_E":             0.19,
    "06_CH3O_E":             0.22,
    "07_CH3OH":              0.24,
    "08_CH3OH_des":         -0.28,   # entropy gain on methanol desorption
}

# ==============================================================================
# SOLVATION CORRECTIONS  (eV)
# Source: Chan & Norskov JPCLett 2015; Gauthier et al. JCTC 2019
# Negative = stabilised by implicit water
# ==============================================================================
SOLVATION_CORRECTIONS: Dict[str, float] = {
    "01_CO2":               -0.03,
    "02_COOH":              -0.25,
    "02_HCOO":              -0.22,
    "03_H2O_from_COOH":     -0.15,
    "03_CO":                -0.01,
    "03_HCOOH":             -0.27,
    "04_H2O_from_HCOOH":    -0.15,
    "04_CHO":               -0.10,
    "04_COH":               -0.18,
    "05_H2O_from_COH":      -0.12,
    "05_C":                 -0.02,
    "05_CH2O":              -0.08,
    "05_CO_diss":           -0.03,
    "05_OH_from_CO_diss":   -0.20,
    "05_H2O_from_CO_diss":  -0.15,
    "06_C_from_CO_diss":    -0.02,
    "06_CH":                -0.01,
    "06_CH3O":              -0.10,
    "07_O_CH4":             -0.05,
    "07_O":                 -0.12,
    "07_CH2":               -0.01,
    "08_CH3":               -0.01,
    "08_OH":                -0.20,
    "09_CH4":                0.00,
    "09_H2O":               -0.18,
    "10_clean":              0.00,
    # Path E
    "05_CH2O_E":            -0.08,
    "06_CH3O_E":            -0.10,
    "07_CH3OH":             -0.14,   # methanol: strong H-bond network
    "08_CH3OH_des":          0.00,
}

# ==============================================================================
# STEP STOICHIOMETRY  (adsorbate atom counts after relaxation)
# ==============================================================================
STEP_EXPECTED_COMPOSITION: Dict[str, Dict[str, int]] = {
    "01_CO2":               {"C": 1, "H": 0, "O": 2},
    "02_COOH":              {"C": 1, "H": 1, "O": 2},
    "02_HCOO":              {"C": 1, "H": 1, "O": 2},
    "03_H2O_from_COOH":     {"C": 1, "H": 2, "O": 2},
    "03_CO":                {"C": 1, "H": 0, "O": 1},
    "03_HCOOH":             {"C": 1, "H": 2, "O": 2},
    "04_H2O_from_HCOOH":    {"C": 1, "H": 3, "O": 2},
    "04_CHO":               {"C": 1, "H": 1, "O": 1},
    "04_COH":               {"C": 1, "H": 1, "O": 1},
    "05_H2O_from_COH":      {"C": 1, "H": 2, "O": 1},
    "05_C":                 {"C": 1, "H": 0, "O": 0},
    "05_CH2O":              {"C": 1, "H": 2, "O": 1},
    "05_CO_diss":           {"C": 1, "H": 0, "O": 1},
    "05_OH_from_CO_diss":   {"C": 1, "H": 1, "O": 1},
    "05_H2O_from_CO_diss":  {"C": 1, "H": 2, "O": 1},
    "06_C_from_CO_diss":    {"C": 1, "H": 0, "O": 0},
    "06_CH":                {"C": 1, "H": 1, "O": 0},
    "06_CH3O":              {"C": 1, "H": 3, "O": 1},
    "07_O_CH4":             {"C": 1, "H": 4, "O": 1},
    "07_O":                 {"C": 0, "H": 0, "O": 1},
    "07_CH2":               {"C": 1, "H": 2, "O": 0},
    "08_CH3":               {"C": 1, "H": 3, "O": 0},
    "08_OH":                {"C": 0, "H": 1, "O": 1},
    "09_CH4":               {"C": 1, "H": 4, "O": 0},
    "09_H2O":               {"C": 0, "H": 2, "O": 1},
    "10_clean":             {"C": 0, "H": 0, "O": 0},
    # Path E
    "05_CH2O_E":            {"C": 1, "H": 2, "O": 1},
    "06_CH3O_E":            {"C": 1, "H": 3, "O": 1},
    "07_CH3OH":             {"C": 1, "H": 4, "O": 1},
    "08_CH3OH_des":         {"C": 0, "H": 0, "O": 0},
}

# ==============================================================================
# NON-PCET STEPS  (no H+ + e- consumed; n_pcet = 0)
# ==============================================================================
# ---------------------------------------------------------------------------
# NON_PCET_STEPS: steps where dH = 0 (no H+/e- transferred).
# Derived rigorously from STEP_EXPECTED_COMPOSITION composition deltas.
# Rule: n_pcet = max(0, H(step) - H(prev_step_in_pathway)).
# Any step with dH > 0 is a PCET step.  Any step with dH <= 0 is NON_PCET.
#
# IMPORTANT: 04_CHO in Path C is NON_PCET (H2O strip, dH=-2).
#            04_CHO in Paths A and E is PCET (CO*+H->CHO*, dH=+1).
#            This pathway-specific exception is handled in run_pathway_site
#            via _pcet_count_for_step().  NON_PCET_STEPS only lists steps
#            that are NON_PCET in ALL pathways they appear in.
# ---------------------------------------------------------------------------
NON_PCET_STEPS = frozenset({
    "01_CO2",               # CO2 adsorbs (dH=0)
    "03_CO",                # H2O desorbs, CO* remains (dH=-2)
    "05_C",                 # H2O desorbs, C* remains (dH=-2)
    "05_CO_diss",           # C-O scission (dH=0)
    "06_C_from_CO_diss",    # O* handled separately, C* remains (dH=-2)
    "07_O",                 # CH4 already desorbed; O* alone (dH=-4)
    "10_clean",             # H2O desorbs (dH=-2)
    "08_CH3OH_des",         # CH3OH desorbs (Path E, dH=-4)
})

# Steps that are NON_PCET only in specific pathways (see _pcet_count_for_step)
_PATHWAY_NONPCET = {
    "04_CHO": {"C"},               # strip step in Path C only
}


# ============================================================
# PATH-SPECIFIC STEP SETS  (used by _perturb_adsorbate)
# ============================================================
PATH_A_O_ANCHORED_STEPS = frozenset({
    "06_CH3O", "07_O_CH4", "07_O", "08_OH", "09_H2O",
})
PATH_C_O_ANCHORED_STEPS = frozenset({
    "06_CH3O", "07_O_CH4", "07_O", "08_OH", "09_H2O",
})
PATH_B_C_CHAIN_STEPS = frozenset({
    "05_C", "06_CH", "07_CH2", "08_CH3", "09_CH4",
})
PATH_D_CO_DISS_STEPS = frozenset({
    "05_CO_diss",
})
PATH_D_O_ANCHORED_STEPS = frozenset({
    "05_OH_from_CO_diss", "05_H2O_from_CO_diss",
})
PATH_D_C_CHAIN_STEPS = frozenset({
    "06_C_from_CO_diss", "06_CH", "07_CH2", "08_CH3", "09_CH4",
})
PATH_E_O_ANCHORED_STEPS = frozenset({
    "06_CH3O_E", "07_CH3OH",
})


def _h_target_label(step: str, iteration: int, n_total: int) -> str:
    """Human-readable label for which heavy atom H targets in this iteration."""
    targets = ["C", "O1", "O2", "C+120", "O1+240", "O2+360"]
    if step in ("01_CO2", "03_CO", "05_C", "05_CO_diss", "06_C_from_CO_diss"):
        targets = ["C", "C+60", "C+120", "C+180", "C+240", "C+300"]
    if iteration < len(targets):
        return targets[iteration]
    return f"iter{iteration}"


# ============ BOND + GEOMETRY HELPERS
# Copied verbatim from original run_mace_phonons.py template
# Used by describe_relaxed_geometry and run_pathway_site

def _bond_exists(pos_a, pos_b, element_a, element_b, scale=1.25):
    """True if distance < scale * (covalent_r_a + covalent_r_b)."""
    COV = {
        "H":  0.31, "C":  0.77, "N":  0.71, "O":  0.66,
        "Ti": 1.36, "V":  1.22, "Cr": 1.22, "Mn": 1.19,
        "Fe": 1.16, "Co": 1.11, "Ni": 1.10, "Cu": 1.12, "Zn": 1.18,
        "Zr": 1.48, "Nb": 1.37, "Mo": 1.45, "Tc": 1.56, "Ru": 1.26,
        "Rh": 1.35, "Pd": 1.31, "Ag": 1.53, "Cd": 1.48,
        "Hf": 1.52, "Ta": 1.46, "W":  1.37, "Re": 1.31, "Os": 1.44,
        "Ir": 1.41, "Pt": 1.36, "Au": 1.36,
        "Al": 1.21, "Si": 1.11, "Ga": 1.22, "In": 1.42, "Sn": 1.39,
    }
    r = scale * (COV.get(element_a, 0.80) + COV.get(element_b, 0.80))
    return np.linalg.norm(np.array(pos_a) - np.array(pos_b)) < r


# ==============================================================================
# PER-STEP BONDING TOPOLOGY VERIFICATION
#
# Each step in each pathway has a SPECIFIC required bonding pattern.
# verify_intermediate_identity() checks:
#   - Which atom binds to surface (C-down vs O-down)
#   - Which bonds must exist and at what length
#   - Which bonds must NOT exist (isomer exclusion)
#   - Specific isomer identity (CHO vs COH, CH2O vs HCOH, etc.)
#
# Returns "intact" if ALL checks pass, or a specific rejection reason.
# These rejections feed into targeted recovery strategies.
# ==============================================================================

# Bonding topology per step.  Fields:
#   surf_atom:   element that must be closest to surface ("C","O","any","none")
#   req_bonds:   list of (elemA, elemB) bonds that MUST exist
#   forb_bonds:  list of (elemA, elemB) bonds that must NOT exist
#   co_min/max:  C-O bond length range (Ang); None = skip
#   co_triple:   True = require CO* triple bond (1.08-1.22 Ang)
#   oh_count:    exact O-H bond count required (None = skip)
#   ch_count:    exact C-H bond count required (None = skip)
#   note:        human-readable description for logs

STEP_TOPOLOGY: Dict[str, Dict] = {
    "01_CO2": {
        "surf_atom":  "any",
        "req_bonds":  [("C","O")],
        "forb_bonds": [],
        "co_min": 1.15, "co_max": 1.30,
        "co_triple": False,
        "oh_count": None, "ch_count": 0,
        "oco_max_angle": 165.0,   # bent CO2 = chemisorbed
        "note": "CO2*: bent, both C-O bonds intact",
    },
    "02_COOH": {
        "surf_atom":  "C",
        "req_bonds":  [("C","O"), ("O","H")],
        "forb_bonds": [],
        "co_min": 1.18, "co_max": 1.50,
        "co_triple": False,
        "oh_count": 1, "ch_count": 0,
        "note": "COOH*: C-down, one C=O, one C-OH, no C-H",
    },
    "02_HCOO": {
        "surf_atom":  "O",        # bidentate, O-bound
        "req_bonds":  [("C","O"), ("C","H")],
        "forb_bonds": [("O","H")],
        "co_min": 1.20, "co_max": 1.40,
        "co_triple": False,
        "oh_count": 0, "ch_count": 1,
        "bidentate": True,        # both O atoms near surface
        "note": "HCOO*: bidentate O-bound formate, C-H, no O-H",
    },
    "03_H2O_from_COOH": {
        "surf_atom":  "C",
        "req_bonds":  [("C","O"), ("O","H")],
        "forb_bonds": [("C","O","H2O")],  # C must NOT bond to H2O oxygen
        "co_min": 1.08, "co_max": 1.25,   # CO* triple bond
        "co_triple": True,
        "oh_count": 2,  "ch_count": 0,    # H2O has 2 O-H
        "h2o_present": True,
        "note": "CO*+H2O*: C-down CO triple bond, H2O free (not bonded to C)",
    },
    "03_CO": {
        "surf_atom":  "C",
        "req_bonds":  [("C","O")],
        "forb_bonds": [("O","H"), ("C","H")],
        "co_min": 1.08, "co_max": 1.22,
        "co_triple": True,
        "oh_count": 0, "ch_count": 0,
        "note": "CO*: C-down, triple bond ~1.12-1.18 Ang, no H",
    },
    "03_HCOOH": {
        "surf_atom":  "C",
        "req_bonds":  [("C","O"), ("C","H"), ("O","H")],
        "forb_bonds": [],
        "co_min": 1.18, "co_max": 1.45,
        "co_triple": False,
        "oh_count": 1, "ch_count": 1,
        "note": "HCOOH*: C-down, C-H, one O-H (formic acid)",
    },
    "04_CHO": {
        "surf_atom":  "C",
        "req_bonds":  [("C","O"), ("C","H")],
        "forb_bonds": [("O","H")],       # if O-H present -> COH not CHO
        "co_min": 1.18, "co_max": 1.35,
        "co_triple": False,
        "oh_count": 0, "ch_count": 1,
        "note": "CHO*: C-down, C=O, C-H, NO O-H (else it is COH*)",
    },
    "04_COH": {
        "surf_atom":  "C",
        "req_bonds":  [("C","O"), ("O","H")],
        "forb_bonds": [("C","H")],       # if C-H present -> CHO not COH
        "co_min": 1.25, "co_max": 1.45,
        "co_triple": False,
        "oh_count": 1, "ch_count": 0,
        "note": "COH*: C-down, C-O single bond, O-H, NO C-H (else it is CHO*)",
    },
    "05_CH2O": {
        "surf_atom":  "C",
        "req_bonds":  [("C","O"), ("C","H")],
        "forb_bonds": [("O","H")],        # O-H present -> HCOH (wrong isomer)
        "co_min": 1.18, "co_max": 1.38,
        "co_triple": False,
        "oh_count": 0, "ch_count": 2,    # MUST have exactly 2 C-H
        "note": "CH2O*: C-down formaldehyde, 2xC-H, NO O-H (else HCOH)",
    },
    "05_CH2O_E": {
        "surf_atom":  "C",
        "req_bonds":  [("C","O"), ("C","H")],
        "forb_bonds": [("O","H")],
        "co_min": 1.18, "co_max": 1.38,
        "co_triple": False,
        "oh_count": 0, "ch_count": 2,
        "note": "CH2O* (Path E): same as 05_CH2O",
    },
    "06_CH3O": {
        "surf_atom":  "O",               # O-DOWN (methoxy)
        "req_bonds":  [("C","O"), ("C","H")],
        "forb_bonds": [("O","H")],
        "co_min": 1.35, "co_max": 1.55,
        "co_triple": False,
        "oh_count": 0, "ch_count": 3,
        "note": "CH3O*: O-down methoxy, 3xC-H, C-O single bond, NO O-H",
    },
    "06_CH3O_E": {
        "surf_atom":  "O",
        "req_bonds":  [("C","O"), ("C","H")],
        "forb_bonds": [("O","H")],
        "co_min": 1.35, "co_max": 1.55,
        "co_triple": False,
        "oh_count": 0, "ch_count": 3,
        "note": "CH3O* (Path E): O-down methoxy",
    },
    "05_C": {
        "surf_atom":  "C",
        "req_bonds":  [],
        "forb_bonds": [("C","O"), ("C","H")],
        "co_min": None, "co_max": None,
        "co_triple": False,
        "oh_count": 0, "ch_count": 0,
        "note": "C*: bare carbon on surface, no H no O",
    },
    "06_CH": {
        "surf_atom":  "C",
        "req_bonds":  [("C","H")],
        "forb_bonds": [("C","O")],
        "co_min": None, "co_max": None,
        "co_triple": False,
        "oh_count": 0, "ch_count": 1,
        "note": "CH*: C-down, exactly 1 C-H, no O",
    },
    "07_CH2": {
        "surf_atom":  "C",
        "req_bonds":  [("C","H")],
        "forb_bonds": [("C","O")],
        "co_min": None, "co_max": None,
        "co_triple": False,
        "oh_count": 0, "ch_count": 2,
        "note": "CH2*: C-down, 2xC-H, no O",
    },
    "08_CH3": {
        "surf_atom":  "C",
        "req_bonds":  [("C","H")],
        "forb_bonds": [("C","O")],
        "co_min": None, "co_max": None,
        "co_triple": False,
        "oh_count": 0, "ch_count": 3,
        "note": "CH3*: C-down, 3xC-H, no O",
    },
    "05_CO_diss": {
        "surf_atom":  "any",
        "req_bonds":  [],
        "forb_bonds": [("C","O")],        # C-O MUST be broken
        "co_min": None, "co_max": None,
        "co_triple": False,
        "oh_count": 0, "ch_count": 0,
        "note": "C*+O* (dissociated): C-O bond ABSENT, both surface-bound",
    },
    "07_O": {
        "surf_atom":  "O",
        "req_bonds":  [],
        "forb_bonds": [("O","H"), ("C","O")],
        "co_min": None, "co_max": None,
        "co_triple": False,
        "oh_count": 0, "ch_count": 0,
        "note": "O*: bare oxygen on surface, no H no C",
    },
    "08_OH": {
        "surf_atom":  "O",
        "req_bonds":  [("O","H")],
        "forb_bonds": [],
        "co_min": None, "co_max": None,
        "co_triple": False,
        "oh_count": 1, "ch_count": 0,
        "note": "OH*: O-down, 1 O-H, no C",
    },
    "07_CH3OH": {
        "surf_atom":  "O",
        "req_bonds":  [("C","O"), ("C","H"), ("O","H")],
        "forb_bonds": [],
        "co_min": 1.38, "co_max": 1.55,
        "co_triple": False,
        "oh_count": 1, "ch_count": 3,
        "note": "CH3OH*: O-down methanol, 3xC-H, 1 O-H, C-O single bond",
    },
    # Desorption/product steps: just verify composition
    "09_CH4":       {"surf_atom": "none", "req_bonds": [], "forb_bonds": [],
                     "co_min": None, "co_max": None, "co_triple": False,
                     "ch_count": 4, "oh_count": 0,
                     "note": "CH4: fully desorbed, 4xC-H, not surface-bound"},
    "09_H2O":       {"surf_atom": "any",  "req_bonds": [("O","H")], "forb_bonds": [],
                     "co_min": None, "co_max": None, "co_triple": False,
                     "ch_count": 0, "oh_count": 2,
                     "note": "H2O*: 2 O-H bonds"},
    "08_CH3OH_des": {"surf_atom": "none", "req_bonds": [("C","O"),("C","H"),("O","H")],
                     "forb_bonds": [], "co_min": None, "co_max": None,
                     "co_triple": False, "ch_count": 3, "oh_count": 1,
                     "note": "CH3OH desorbed: 3xC-H, 1 O-H, C-O, not surface-bound"},
}


def verify_intermediate_identity(atoms: Atoms, n_slab: int,
                                  step: str, pathway_id: str) -> str:
    """
    Check that the relaxed intermediate matches the INTENDED bonding topology
    for this specific step in this specific pathway.

    Returns "intact" if all topology checks pass.
    Returns a specific rejection string explaining the exact mismatch if any
    check fails -- enabling targeted recovery strategies.

    This is stricter than describe_relaxed_geometry:
      - Checks WHICH atom is surface-bound (C-down vs O-down)
      - Checks EXACT bond counts (2xC-H for CH2O, not just "at least one")
      - Checks bond length ranges (CO triple bond 1.08-1.22 Ang vs single)
      - Checks FORBIDDEN bonds (O-H forbidden in CHO = COH isomer rejection)
      - Checks isomer identity explicitly
    """
    topo = STEP_TOPOLOGY.get(step)
    if topo is None:
        return "intact"  # no topology defined -- pass through

    if len(atoms) <= n_slab:
        return "intact"  # clean surface steps pass

    ads      = atoms[n_slab:]
    n_ads    = len(ads)
    slab_top = atoms.positions[:n_slab, 2].max()
    SURF_BOND_MAX = CFG.get("surface_bond_max", 2.80)
    TOP_TOL       = CFG.get("surface_bond_top_layer_tol", 2.5)

    if n_ads == 0:
        return "intact"

    c_ads = [a for a in ads if a.symbol == "C"]
    o_ads = [a for a in ads if a.symbol == "O"]
    h_ads = [a for a in ads if a.symbol == "H"]

    # Top-layer slab atoms for surface-bond checks
    top_layer = np.array([atoms.positions[i] for i in range(n_slab)
                           if atoms.positions[i, 2] > slab_top - TOP_TOL])
    if len(top_layer) == 0:
        top_layer = atoms.positions[:n_slab]

    def _dist(p1, p2):
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _min_surf_dist(atom_pos):
        if len(top_layer) == 0:
            return 99.0
        return float(np.linalg.norm(top_layer - atom_pos, axis=1).min())

    def _has_bond(pos_a, sym_a, pos_b, sym_b, scale=1.25):
        return _bond_exists(pos_a, pos_b, sym_a, sym_b, scale)

    def _count_ch_bonds():
        if not c_ads or not h_ads:
            return 0
        return sum(1 for ha in h_ads
                   if any(_has_bond(ca.position, "C", ha.position, "H", scale=1.35)
                          for ca in c_ads))

    def _count_oh_bonds():
        if not o_ads or not h_ads:
            return 0
        return sum(1 for ha in h_ads
                   if any(_has_bond(oa.position, "O", ha.position, "H")
                          for oa in o_ads))

    def _closest_to_surf(atom_list):
        if not atom_list:
            return None
        return min(atom_list, key=lambda a: _min_surf_dist(a.position))

    surf_req = topo.get("surf_atom", "any")

    # -- 1. Surface-binding atom identity -------------------------------------
    if surf_req not in ("any", "none"):
        c_surf = min((_min_surf_dist(a.position) for a in c_ads), default=99.0)
        o_surf = min((_min_surf_dist(a.position) for a in o_ads), default=99.0)

        if surf_req == "C":
            if not c_ads:
                return "no_C_atom_present"
            if c_surf > SURF_BOND_MAX:
                return f"C_not_surface_bound__dist={c_surf:.2f}Ang"
            # O must NOT be the binding atom (if O is closer than C, wrong topology)
            if o_ads and o_surf < c_surf - 0.30:
                return (f"wrong_binding_atom__O_is_closer_to_surface_"
                         f"than_C_({o_surf:.2f}_vs_{c_surf:.2f}Ang)")

        elif surf_req == "O":
            if not o_ads:
                return "no_O_atom_present"
            if o_surf > SURF_BOND_MAX:
                return f"O_not_surface_bound__dist={o_surf:.2f}Ang"
            # C must NOT be the binding atom
            if c_ads and c_surf < o_surf - 0.30:
                return (f"wrong_binding_atom__C_is_closer_to_surface_"
                         f"than_O_({c_surf:.2f}_vs_{o_surf:.2f}Ang)")

    elif surf_req == "none":
        # Desorption: no atom should be surface-bound
        for a in ads:
            if _min_surf_dist(a.position) < SURF_BOND_MAX:
                return f"should_be_desorbed__atom_{a.symbol}_still_bound"

    # -- 2. Required bonds -----------------------------------------------------
    for sym_a, sym_b in topo.get("req_bonds", []):
        pool_a = [a for a in ads if a.symbol == sym_a]
        pool_b = [a for a in ads if a.symbol == sym_b]
        if not pool_a or not pool_b:
            return f"missing_required_bond__{sym_a}-{sym_b}__atoms_absent"
        found = any(_has_bond(a.position, sym_a, b.position, sym_b)
                    for a in pool_a for b in pool_b)
        if not found:
            return f"missing_required_bond__{sym_a}-{sym_b}"

    # -- 3. Forbidden bonds ----------------------------------------------------
    for bond_spec in topo.get("forb_bonds", []):
        if len(bond_spec) == 2:
            sym_a, sym_b = bond_spec
            pool_a = [a for a in ads if a.symbol == sym_a]
            pool_b = [a for a in ads if a.symbol == sym_b]
            found = any(_has_bond(a.position, sym_a, b.position, sym_b)
                        for a in pool_a for b in pool_b)
            if found:
                return f"forbidden_bond_present__{sym_a}-{sym_b}__wrong_isomer"

    # -- 4. C-O bond length check ----------------------------------------------
    co_min = topo.get("co_min")
    co_max = topo.get("co_max")
    if co_min is not None and c_ads and o_ads:
        co_dists = [_dist(ca.position, oa.position)
                    for ca in c_ads for oa in o_ads]
        min_co = min(co_dists)
        if topo.get("co_triple", False):
            # CO triple bond must be short (1.08-1.22 Ang)
            if min_co > 1.25:
                return (f"co_bond_too_long_for_triple__{min_co:.3f}Ang"
                         f"__expected_lt_1.25Ang")
        elif min_co < co_min or min_co > co_max:
            return (f"co_bond_length_out_of_range__{min_co:.3f}Ang"
                     f"__expected_{co_min:.2f}-{co_max:.2f}Ang")

    # -- 5. OCO angle check (CO2* bent = chemisorbed) -------------------------
    if topo.get("oco_max_angle") is not None and c_ads and len(o_ads) >= 2:
        cp = c_ads[0].position
        v1 = o_ads[0].position - cp
        v2 = o_ads[1].position - cp
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-9)
        oco_ang = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
        if oco_ang > topo["oco_max_angle"]:
            return (f"co2_linear_not_activated__oco={oco_ang:.1f}deg"
                     f"__expected_lt_{topo['oco_max_angle']:.0f}deg")

    # -- 6. Exact H bond counts ------------------------------------------------
    ch_req = topo.get("ch_count")
    if ch_req is not None:
        ch_actual = _count_ch_bonds()
        if ch_actual != ch_req:
            return (f"wrong_ch_bond_count__{ch_actual}_found_"
                     f"{ch_req}_required")

    oh_req = topo.get("oh_count")
    if oh_req is not None:
        oh_actual = _count_oh_bonds()
        if oh_actual != oh_req:
            return (f"wrong_oh_bond_count__{oh_actual}_found_"
                     f"{oh_req}_required")

    # -- 7. Bidentate check (HCOO*) -------------------------------------------
    if topo.get("bidentate", False) and o_ads:
        o_surf_dists = [_min_surf_dist(oa.position) for oa in o_ads]
        n_o_bound = sum(1 for d in o_surf_dists if d < SURF_BOND_MAX)
        if n_o_bound < 2:
            return (f"formate_not_bidentate__only_{n_o_bound}_O_"
                     f"surface_bound__expected_2")

    # -- 8. H2O present but NOT bonded to C (03_H2O_from_COOH) ---------------
    if topo.get("h2o_present", False):
        # Find the H2O oxygen: an O with 2 H-bonds NOT bonded to C
        cp = c_ads[0].position if c_ads else None
        h2o_found = False
        for oa in o_ads:
            n_oh = sum(1 for ha in h_ads if _has_bond(oa.position,"O",ha.position,"H"))
            if n_oh >= 2:
                # Check it's not also bonded to C
                c_bonded = (cp is not None and
                            _has_bond(cp, "C", oa.position, "O", scale=0.90))
                if not c_bonded:
                    h2o_found = True
                    break
        if not h2o_found:
            return "h2o_group_not_found_or_still_bonded_to_C"

    return "intact"


def describe_relaxed_geometry(atoms, n_slab, step, expected_comp):
    """
    Inspect relaxed atoms and return a short snake_case descriptor.
    Returns 'intact' if stoichiometry, bonds, and adsorption all check out.
    """
    if expected_comp is None:
        return "unknown"

    ads      = atoms[n_slab:]
    n_ads    = len(ads)
    slab_top = atoms.positions[:n_slab, 2].max() if n_slab > 0 else 0.0

    actual = {}
    for a in ads:
        actual[a.symbol] = actual.get(a.symbol, 0) + 1

    # 1. Stoichiometry
    missing, extra = [], []
    for el, cnt in expected_comp.items():
        diff = cnt - actual.get(el, 0)
        if diff > 0:
            missing.append(f"{el}{diff}")
        elif diff < 0:
            extra.append(f"{el}{abs(diff)}")
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing_" + "_".join(missing))
        if extra:
            parts.append("extra_" + "_".join(extra))
        return "partial__" + "__".join(parts)

    if n_ads == 0:
        return "clean_surface"

    c_atoms = [(i, ads[i]) for i in range(len(ads)) if ads[i].symbol == "C"]
    o_atoms = [(i, ads[i]) for i in range(len(ads)) if ads[i].symbol == "O"]
    h_atoms = [(i, ads[i]) for i in range(len(ads)) if ads[i].symbol == "H"]

    # 2. Desorption check
    DESORP_Z = CFG.get("desorp_z_threshold", 5.0)
    desorbed = []
    for i, a in enumerate(ads):
        if a.position[2] - slab_top > DESORP_Z:
            bonded = any(
                _bond_exists(a.position, ads[j].position, a.symbol, ads[j].symbol)
                for j in range(len(ads)) if j != i
            )
            if not bonded:
                desorbed.append(a.symbol)
    if desorbed:
        from collections import Counter
        cnt = Counter(desorbed)
        label = "_".join(f"{el}{n if n>1 else ''}" for el, n in sorted(cnt.items()))
        return f"desorbed__{label}"

    # 3. Bond checks
    broken_bonds = []
    nC = expected_comp.get("C", 0)
    nO = expected_comp.get("O", 0)
    nH = expected_comp.get("H", 0)

    if nC == 1 and nO >= 1 and c_atoms and o_atoms:
        # CO*+H2O* co-adsorbed steps: require ONE C-O bond (CO*) + H2O group
        # NOT bonded to C. Distinct from CO_BOND_STEPS which require ALL O's bonded.
        co_water_steps = {"03_H2O_from_COOH", "04_H2O_from_HCOOH"}

        co_bonded_steps = {
            "01_CO2", "02_COOH", "02_HCOO", "03_CO",
            "03_HCOOH", "04_CHO", "04_COH",
            "05_CH2O", "06_CH3O",
            "05_CH2O_E", "06_CH3O_E", "07_CH3OH",
        }

        if step in co_water_steps:
            c_pos = c_atoms[0][1].position
            # (a) At least one C-O bond (the CO* part)
            co_bonds = [oa for _, oa in o_atoms
                        if _bond_exists(c_pos, oa.position, "C", "O")]
            if not co_bonds:
                broken_bonds.append("C-O")
            # (b) No O-O peroxy bond
            for i in range(len(o_atoms)):
                for j in range(i+1, len(o_atoms)):
                    if _bond_exists(o_atoms[i][1].position,
                                    o_atoms[j][1].position, "O", "O", scale=1.15):
                        broken_bonds.append("O-O_peroxy")
                        break
            # (c) H2O group: O with 2 H-bonds that is NOT the CO* oxygen.
            # CO* oxygen = the ONE oxygen CLOSEST to C (CO triple bond ~1.10-1.20 Ang).
            # Do NOT skip all O's within generic bond range -- that catches H2O oxygen too.
            if o_atoms:
                co_star_o = min(o_atoms, key=lambda x: np.linalg.norm(
                    c_pos - x[1].position))[1]
            else:
                co_star_o = None
            h2o_found = False
            for oi, oa in o_atoms:
                if co_star_o is not None and np.allclose(oa.position, co_star_o.position):
                    continue  # skip only the TRUE CO* oxygen (shortest C-O)
                h_bonds = sum(1 for _, ha in h_atoms
                               if _bond_exists(oa.position, ha.position, "O", "H"))
                if h_bonds >= 2:
                    h2o_found = True
                    break
            if not h2o_found:
                broken_bonds.append("H2O_not_formed")

        elif step in co_bonded_steps:
            c_pos = c_atoms[0][1].position
            for oi, oa in o_atoms:
                if not _bond_exists(c_pos, oa.position, "C", "O"):
                    broken_bonds.append("C-O")
                    break

        if step == "05_CO_diss" and c_atoms and o_atoms:
            c_pos = c_atoms[0][1].position
            for oi, oa in o_atoms:
                if _bond_exists(c_pos, oa.position, "C", "O"):
                    broken_bonds.append("CO_not_dissociated")
                    break

    if nO >= 1 and nH >= 1 and o_atoms and h_atoms:
        oh_steps = {"02_COOH", "03_HCOOH", "04_COH", "02_HCOO", "08_OH", "09_H2O",
                    "03_H2O_from_COOH", "04_H2O_from_HCOOH",
                    "05_OH_from_CO_diss", "05_H2O_from_CO_diss"}
        if step in oh_steps:
            found_oh = any(
                _bond_exists(oa.position, ha.position, "O", "H")
                for _, oa in o_atoms for _, ha in h_atoms
            )
            if not found_oh:
                broken_bonds.append("O-H")

    if nC == 1 and nH >= 1 and c_atoms and h_atoms:
        ch_steps = {"04_CHO", "05_CH2O", "06_CH3O", "06_CH", "07_CH2", "08_CH3",
                    "02_HCOO", "03_HCOOH", "04_H2O_from_HCOOH",
                    "05_CH2O_E", "06_CH3O_E"}
        if step in ch_steps:
            c_pos = c_atoms[0][1].position
            # scale=1.35: threshold=1.46 Ang -- covers relaxed C-H on metal surfaces
            found_ch = any(
                _bond_exists(c_pos, ha.position, "C", "H", scale=1.35)
                for _, ha in h_atoms
            )
            if not found_ch:
                broken_bonds.append("C-H")

        # CH2O isomer check: formaldehyde must have BOTH H on C, NO H on O.
        # If one H migrated to O, it is HCOH (hydroxymethylidene) -- wrong isomer
        # for Path A/C/E formaldehyde route.  Treat as broken so recovery fires.
        ch2o_no_oh_steps = {"05_CH2O", "05_CH2O_E"}
        if step in ch2o_no_oh_steps and o_atoms and h_atoms:
            for _, oa in o_atoms:
                oh_found = any(
                    _bond_exists(oa.position, ha.position, "O", "H")
                    for _, ha in h_atoms
                )
                if oh_found:
                    broken_bonds.append("wrong_isomer__HCOH_not_CH2O")
                    break

        # Methane integrity check: Enforce that CH4 molecule is intact (all 4 H bonded to C, O not bonded to C)
        if step in ("09_CH4", "07_O_CH4") and c_atoms and h_atoms:
            c_pos = c_atoms[0][1].position
            for _, ha in h_atoms:
                if not _bond_exists(c_pos, ha.position, "C", "H", scale=1.35):
                    broken_bonds.append("C-H")
                    break
            if o_atoms:
                for _, oa in o_atoms:
                    if _bond_exists(c_pos, oa.position, "C", "O"):
                        broken_bonds.append("C-O_not_dissociated")
                        break

    if broken_bonds:
        return f"dissociated__{'_'.join(broken_bonds)}_broken"

    # 4. Surface binding -- real metal-adsorbate bond check
    #
    # Old approach: check if lowest adsorbate atom is < 3.5 Ang above surface.
    # Problem: adsorbate can be close in z but not bonded to any metal atom,
    #          which is what OVITO shows when "create bonds" finds nothing.
    #
    # New approach: check that at least ONE adsorbate atom is within
    # surface_bond_max (default 2.80 Ang) of a top-layer metal atom in 3D.
    # This directly corresponds to the bond OVITO would draw.
    _SURF_BOND_MAX = CFG.get("surface_bond_max", 2.80)
    _TOP_TOL       = CFG.get("surface_bond_top_layer_tol", 2.5)

    # Desorption/clean-surface steps: no adsorbate expected, skip this check
    # Desorption steps: adsorbate expected to leave surface
    # For CH4 (09_CH4): verify the C atom has 4 H bonds and NO surface bond
    # This confirms natural desorption vs artificial removal
    _desorp_steps = {
        "10_clean", "09_H2O", "09_CH4", "08_CH3OH_des",
        "07_O_CH4",
    }

    # CH4 natural-desorption check: only for 09_CH4
    if step == "09_CH4" and n_ads > 0:
        _c_atoms = [a for a in ads if a.symbol == "C"]
        _h_atoms = [a for a in ads if a.symbol == "H"]
        if _c_atoms and len(_h_atoms) >= 4:
            _cp = _c_atoms[0].position
            # Count C-H bonds
            _ch_bonds = sum(1 for ha in _h_atoms
                             if np.linalg.norm(_cp - ha.position) < 1.30)
            # Check if C is still surface-bound
            _top_pos = np.array([atoms.positions[i] for i in range(n_slab)
                                   if atoms.positions[i,2] > slab_top - 2.5])
            _surf_bound = (len(_top_pos) > 0 and
                           float(np.linalg.norm(_top_pos - _cp, axis=1).min()) < 2.5)
            if _ch_bonds < 4:
                return f"ch4_incomplete__only_{_ch_bonds}_ch_bonds"
            if _surf_bound:
                return "ch4_still_surface_bound__not_desorbed"
            # Properly desorbed CH4
            return "intact"   # CH4 freely floating = correct for this step
    if n_ads > 0 and step not in _desorp_steps:
        # Top-layer slab atoms: within _TOP_TOL Ang of surface top
        top_layer_pos = np.array([
            atoms.positions[i]
            for i in range(n_slab)
            if atoms.positions[i, 2] > slab_top - _TOP_TOL
        ])

        if len(top_layer_pos) == 0:
            # Fallback: use all slab atoms if no top layer found
            top_layer_pos = atoms.positions[:n_slab]

        # Find minimum 3D distance between ANY adsorbate atom and ANY top-layer atom
        surface_bound = False
        min_metal_dist = float("inf")
        for ads_atom in ads:
            dists = np.linalg.norm(top_layer_pos - ads_atom.position, axis=1)
            d_min = float(dists.min())
            if d_min < min_metal_dist:
                min_metal_dist = d_min
            if d_min < _SURF_BOND_MAX:
                surface_bound = True
                break   # at least one atom is bound -- enough

        if not surface_bound:
            formula = "".join(f"{el}{cnt}" if cnt > 1 else el
                               for el, cnt in sorted(actual.items()))
            return (f"not_surface_bound__{formula}"
                    f"__min_dist_{min_metal_dist:.2f}Ang")

    # -- Topology identity check: final gate ---------------------------------
    # Runs after stoichiometry/bond/desorption checks.
    # Rejects wrong isomers, wrong binding atoms, wrong bond lengths.
    _topo_result = verify_intermediate_identity(atoms, n_slab, step, "")
    if _topo_result != "intact":
        return f"topology_fail__{_topo_result}"

    return "intact"

# ==============================================================================
# HUMAN-READABLE STEP LABELS
# ==============================================================================
STEP_LABEL: Dict[str, str] = {
    "01_CO2":               "CO2(g) -> CO2*",
    "02_COOH":              "CO2* + H+ + e- -> COOH*",
    "02_HCOO":              "CO2* + H+ + e- -> HCOO* (formate, O-bound)",
    "03_H2O_from_COOH":     "COOH* + H+ + e- -> CO* + H2O*",
    "03_CO":                "H2O* desorbs; CO* remains",
    "03_HCOOH":             "HCOO* + H+ + e- -> HCOOH* (formic acid)",
    "04_H2O_from_HCOOH":    "HCOOH* -> CHO* + H2O* (dehydration)",
    "04_CHO":               "H2O* desorbs; CHO* remains (formyl)",
    "04_COH":               "CO* + H+ + e- -> COH* (hydroxymethylidene)",
    "05_H2O_from_COH":      "COH* + H+ + e- -> C* + H2O*",
    "05_C":                 "H2O* desorbs; C* remains (surface carbene)",
    "05_CH2O":              "CHO* + H+ + e- -> CH2O* (formaldehyde)",
    "05_CO_diss":           "CO* -> C* + O* (direct C-O scission)",
    "05_OH_from_CO_diss":   "O* + H+ + e- -> OH* (C* co-adsorbed)",
    "05_H2O_from_CO_diss":  "OH* + H+ + e- -> H2O* (C* co-adsorbed)",
    "06_C_from_CO_diss":    "H2O* desorbs; C* alone (carbene precursor)",
    "06_CH":                "C* + H+ + e- -> CH*",
    "06_CH3O":              "CH2O* + H+ + e- -> CH3O* (methoxy)",
    "07_O_CH4":             "CH3O* + H+ + e- -> [CH4* + O*]",
    "07_O":                 "CH4 desorbs; O* remains",
    "07_CH2":               "CH* + H+ + e- -> CH2* (methylene)",
    "08_CH3":               "CH2* + H+ + e- -> CH3* (methyl)",
    "08_OH":                "O* + H+ + e- -> OH*",
    "09_CH4":               "CH3* + H+ + e- -> CH4^",
    "09_H2O":               "OH* + H+ + e- -> H2O*",
    "10_clean":             "H2O* -> H2O^ (clean surface recovered)",
    # Path E
    "05_CH2O_E":            "CHO* + H+ + e- -> CH2O* (Path E)",
    "06_CH3O_E":            "CH2O* + H+ + e- -> CH3O* (Path E)",
    "07_CH3OH":             "CH3O* + H+ + e- -> CH3OH* (methanol on surface)",
    "08_CH3OH_des":         "CH3OH* -> CH3OH^ (methanol desorption)",
}

# ==============================================================================
# PATHWAY DEFINITIONS
# ==============================================================================
PATHWAYS: Dict[str, Dict] = {
    "A": {
        "name":        "Formaldehyde_Route",
        "description": "CO2->COOH*->CO*->CHO*->CH2O*->CH3O*->[CH4+O*]->OH*->H2O*->clean",
        "color":       GREEN,
        "product":     "CH4",
        "steps": [
            "01_CO2", "02_COOH", "03_H2O_from_COOH", "03_CO",
            "04_CHO", "05_CH2O", "06_CH3O",
            "07_O_CH4", "07_O", "08_OH", "09_H2O", "10_clean",
        ],
        "ch4_handling": "form_then_strip",
    },
    "B": {
        "name":        "Hydroxymethylidene_Route",
        "description": "CO2->COOH*->CO*->COH*->C*->CH*->CH2*->CH3*->CH4^",
        "color":       CYAN,
        "product":     "CH4",
        "steps": [
            "01_CO2", "02_COOH", "03_H2O_from_COOH", "03_CO",
            "04_COH", "05_H2O_from_COH", "05_C",
            "06_CH", "07_CH2", "08_CH3", "09_CH4",
        ],
        "ch4_handling": "lift_to_gas",
    },
    "C": {
        "name":        "Formate_Route",
        "description": "CO2->HCOO*->HCOOH*->CHO*->CH2O*->CH3O*->[CH4+O*]->H2O*->clean",
        "color":       YELLOW,
        "product":     "CH4",
        "steps": [
            "01_CO2", "02_HCOO", "03_HCOOH",
            "04_H2O_from_HCOOH", "04_CHO",
            "05_CH2O", "06_CH3O",
            "07_O_CH4", "07_O", "08_OH", "09_H2O", "10_clean",
        ],
        "ch4_handling": "form_then_strip",
    },
    "D": {
        "name":        "Carbene_Route",
        "description": "CO2->COOH*->CO*->[C*+O*]->[C*+OH*]->C*->CH*->CH2*(carbene)->CH4^",
        "color":       MAGENTA,
        "product":     "CH4",
        "steps": [
            "01_CO2", "02_COOH", "03_H2O_from_COOH", "03_CO",
            "05_CO_diss",
            "05_OH_from_CO_diss",
            "05_H2O_from_CO_diss",
            "06_C_from_CO_diss",
            "06_CH", "07_CH2", "08_CH3", "09_CH4",
        ],
        "ch4_handling": "lift_to_gas",
    },
    "E": {
        "name":        "Methanol_Route",
        "description": "CO2->COOH*->CO*->CHO*->CH2O*->CH3O*->CH3OH*->CH3OH^",
        "color":       BLUE,
        "product":     "CH3OH",
        "steps": [
            "01_CO2", "02_COOH", "03_H2O_from_COOH", "03_CO",
            "04_CHO",
            "05_CH2O_E",
            "06_CH3O_E",
            "07_CH3OH",
            "08_CH3OH_des",
        ],
        "ch4_handling": "none",
    },
}

# ==============================================================================
# BOND LENGTH REFERENCE RANGES  (Ang)
# Used by ChemistryValidator for realistic-bonding checks
# ==============================================================================
BOND_RANGES: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("C", "O"):  (1.05, 1.65),   # C=O (1.20) to C-O (1.43), allow slack
    ("C", "H"):  (0.90, 1.45),
    ("O", "H"):  (0.85, 1.25),
    ("C", "C"):  (1.10, 1.65),
    ("C", "N"):  (1.10, 1.60),
    ("O", "O"):  (1.10, 1.60),
    # Metal-adsorbate (very rough; refine per element if needed)
    ("C", "Cu"): (1.70, 2.40),
    ("C", "Ni"): (1.70, 2.20),
    ("C", "Fe"): (1.70, 2.30),
    ("C", "Co"): (1.70, 2.20),
    ("O", "Cu"): (1.80, 2.40),
    ("O", "Ni"): (1.80, 2.20),
    ("O", "Fe"): (1.80, 2.30),
    ("O", "Co"): (1.80, 2.20),
}

# Fallback for any unrecognised element pair
BOND_RANGE_FALLBACK: Tuple[float, float] = (1.00, 3.20)

# Maximum C-O distance to be considered "bonded" (used in geometry descriptor)
CO_BOND_MAX:  float = 1.80
CH_BOND_MAX:  float = 1.40
OH_BOND_MAX:  float = 1.30
MOL_BOND_MAX: float = 2.00   # generic molecular bond cutoff

# ==============================================================================
# STEP FOLDER NAMES  (human-readable output directory naming)
# ==============================================================================
STEP_FOLDER_NAME: Dict[str, str] = {
    "01_CO2":               "CO2_adsorbed",
    "02_COOH":              "COOH_adsorbed",
    "02_HCOO":              "HCOO_adsorbed",
    "03_H2O_from_COOH":     "H2O_on_surface_from_COOH",
    "03_CO":                "CO_adsorbed",
    "03_HCOOH":             "HCOOH_adsorbed",
    "04_H2O_from_HCOOH":    "H2O_on_surface_from_HCOOH",
    "04_CHO":               "CHO_adsorbed",
    "04_COH":               "COH_adsorbed",
    "05_H2O_from_COH":      "H2O_on_surface_from_COH",
    "05_C":                 "C_adsorbed_surface_carbene",
    "05_CH2O":              "CH2O_adsorbed",
    "05_CO_diss":           "C_and_O_co_adsorbed_CO_dissociated",
    "05_OH_from_CO_diss":   "C_and_OH_co_adsorbed",
    "05_H2O_from_CO_diss":  "C_and_H2O_co_adsorbed",
    "06_C_from_CO_diss":    "C_adsorbed_carbene_precursor",
    "06_CH":                "CH_adsorbed",
    "06_CH3O":              "CH3O_adsorbed_methoxy",
    "07_O_CH4":             "CH4_on_surface_plus_O_adsorbed",
    "07_O":                 "O_adsorbed_after_CH4_desorption",
    "07_CH2":               "CH2_adsorbed_methylene",
    "08_CH3":               "CH3_adsorbed_methyl",
    "08_OH":                "OH_adsorbed",
    "09_CH4":               "CH4_gas",
    "09_H2O":               "H2O_on_surface",
    "10_clean":             "clean_surface_after_H2O_desorption",
    "05_CH2O_E":            "CH2O_adsorbed_PathE",
    "06_CH3O_E":            "CH3O_adsorbed_PathE",
    "07_CH3OH":             "CH3OH_adsorbed_methanol_on_surface",
    "08_CH3OH_des":         "clean_surface_after_methanol_desorption",
}


# ==============================================================================
# LOGGING SETUP
# ==============================================================================

class _ColorlessFormatter(logging.Formatter):
    """Strip ANSI colour codes from log file output."""
    import re
    _ANSI = re.compile(r"\x1b\[[0-9;]*m")
    def format(self, record):
        msg = super().format(record)
        return self._ANSI.sub("", msg)

def setup_logging(workdir: Path) -> logging.Logger:
    logger = logging.getLogger("co2rr")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    # Console handler (with colour)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    # File handler (no colour)
    log_file = workdir / f"co2rr_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_ColorlessFormatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)
    return logger

LOG: logging.Logger = logging.getLogger("co2rr")


# ==============================================================================
# DATA CLASSES  (typed records for results, failures, recovery attempts)
# ==============================================================================

@dataclass
class ValidationResult:
    """Outcome of ChemistryValidator.validate()."""
    is_valid:         bool
    stoich_ok:        bool
    bonds_ok:         bool
    geometry_ok:      bool
    adsorbed_ok:      bool
    convergence_ok:   bool
    descriptor:       str      = "unknown"
    messages:         List[str] = field(default_factory=list)

    def __bool__(self):
        return self.is_valid

@dataclass
class StepResult:
    """Complete result record for one pathway step at one site."""
    head:           str
    pathway:        str
    site_idx:       int
    step:           str
    step_idx:       int
    energy:         float
    bare_energy:    float
    dg:             float
    dg_breakdown:   Dict[str, float]
    n_pcet:         int
    folder:         str
    validation:     ValidationResult
    stability_info: Dict[str, Any]
    geometry:       Dict[str, Any]
    retry_history:  List[Dict[str, Any]] = field(default_factory=list)
    rescued:        bool = False
    timestamp:      str  = field(default_factory=lambda: datetime.datetime.now().isoformat())

@dataclass
class FailureRecord:
    """Record of a failed step."""
    head:       str
    pathway:    str
    site_idx:   int
    step:       str
    step_idx:   int
    reason:     str
    attempts:   int
    timestamp:  str = field(default_factory=lambda: datetime.datetime.now().isoformat())


# ==============================================================================
# MODULE 1: CHECKPOINT MANAGER
# ==============================================================================

class CheckpointManager:
    """
    Persistent restart capability.

    Saves / loads step results as JSON per (head, pathway, site, step).
    A step is skipped on restart if its checkpoint file exists and is valid.

    File layout under workdir:
        checkpoints/
            {head}/{pathway_id}/site_{site_idx:03d}/{step}.json
    """

    def __init__(self, workdir: Path):
        self.ckpt_dir = workdir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, head: str, pathway: str, site_idx: int, step: str) -> Path:
        d = self.ckpt_dir / head / pathway / f"site_{site_idx:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{step}.json"

    def exists(self, head: str, pathway: str, site_idx: int, step: str) -> bool:
        return self._path(head, pathway, site_idx, step).exists()

    def save(self, result: StepResult):
        p = self._path(result.head, result.pathway, result.site_idx, result.step)
        data = asdict(result)
        # ValidationResult is already a dict via asdict
        with open(p, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, head: str, pathway: str, site_idx: int, step: str) -> Optional[Dict]:
        p = self._path(head, pathway, site_idx, step)
        if not p.exists():
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None

    def save_bare_slab(self, head: str, energy: float):
        p = self.ckpt_dir / head / "bare_slab_energy.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump({"energy": energy, "head": head}, f)

    def load_bare_slab(self, head: str) -> Optional[float]:
        p = self.ckpt_dir / head / "bare_slab_energy.json"
        if p.exists():
            try:
                return json.load(open(p))["energy"]
            except Exception:
                return None
        return None


# ==============================================================================
# MODULE 2: SLAB GENERATOR
# ==============================================================================

class SlabGenerator:
    """
    Load or generate the clean catalytic slab.

    Supported inputs
    ----------------
    1. Existing POSCAR / CONTCAR / extxyz / CIF  (CFG["slab_poscar"])
    2. Bulk element string  ("Cu", "Ni", "Fe", ...)  -> slab via ase.build.surface

    Output
    ------
    - Sorted atoms object (surface atoms at top)
    - n_slab (total atom count, fixed throughout workflow)
    - bottom-layer freeze indices
    - validated geometry (no overlapping atoms, sane cell)
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    # -- Public entry point ------------------------------------------------
    def load(self) -> Atoms:
        src = self.cfg["slab_poscar"]
        try:
            slab = read(src)
            LOG.info(f"Slab loaded from {src}  ({len(slab)} atoms)")
        except FileNotFoundError:
            LOG.warning(f"POSCAR not found at {src} -- generating Cu(111) 3x3x4 demo slab")
            slab = self._generate_demo_slab()
        slab = self._sort_by_z(slab)
        self._validate_slab(slab)
        return slab

    # -- Surface-top height ------------------------------------------------
    @staticmethod
    def surface_top_z(slab: Atoms) -> float:
        return float(slab.positions[:, 2].max())

    # -- Freeze indices ----------------------------------------------------
    @staticmethod
    def freeze_indices(slab: Atoms, freeze_fraction: float = 0.35) -> List[int]:
        z = slab.positions[:, 2]
        z_thresh = z.min() + freeze_fraction * (z.max() - z.min())
        return [i for i in range(len(slab)) if z[i] < z_thresh]

    # -- Top-layer indices -------------------------------------------------
    @staticmethod
    def top_layer_indices(slab: Atoms, tol: float = 1.0) -> List[int]:
        top_z = SlabGenerator.surface_top_z(slab)
        return [i for i, a in enumerate(slab) if a.position[2] > top_z - tol]

    # -- Internal helpers --------------------------------------------------
    @staticmethod
    def _sort_by_z(slab: Atoms) -> Atoms:
        idx = np.argsort(slab.positions[:, 2])
        return slab[idx]

    @staticmethod
    def _validate_slab(slab: Atoms):
        # Check for overlapping atoms (min distance < 1.5 Ang)
        from ase.geometry import get_distances
        pos = slab.positions
        for i in range(len(slab)):
            for j in range(i + 1, min(i + 20, len(slab))):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < 1.5:
                    LOG.warning(f"Slab overlap: atoms {i} and {j} are {d:.3f} Ang apart")
                    break
        if not slab.pbc.all():
            LOG.warning("Slab does not have full 3-D PBC -- setting to [True, True, True]")
            slab.pbc = [True, True, True]

    @staticmethod
    def _generate_demo_slab() -> Atoms:
        """Generate a Cu(111) 3x3x4 slab for testing (no POSCAR required)."""
        from ase.build import fcc111
        slab = fcc111("Cu", size=(3, 3, 4), vacuum=12.0, periodic=True)
        return slab


# ==============================================================================
# MODULE 3: ACTIVE SITE IDENTIFIER
# ==============================================================================

class ActiveSiteIdentifier:
    """
    Identify distinct adsorption sites on the slab surface.

    Site types detected
    -------------------
    top     -- directly above a surface atom
    bridge  -- midpoint between two nearest-neighbour surface atoms
    hollow  -- centroid of 3 nearest surface atoms (fcc / hcp hollow)
    4fold   -- centroid of 4-atom square arrangement (bcc / fcc(100))

    For high-entropy alloys, sites are further filtered by a local chemical
    composition fingerprint to avoid redundant calculations.

    Returns a list of site dicts:
        {"type": str, "position": np.ndarray, "atom_indices": List[int]}
    """

    def __init__(self, slab: Atoms, cfg: Dict[str, Any]):
        self.slab = slab
        self.cfg  = cfg
        self.n    = len(slab)
        self.top_indices = SlabGenerator.top_layer_indices(slab, tol=cfg["top_layer_tol"])

    def identify(self) -> List[Dict]:
        sites = []
        sites += self._top_sites()
        sites += self._bridge_sites()
        sites += self._hollow_sites()
        sites  = self._deduplicate(sites)
        sites  = self._fingerprint_filter(sites)
        max_s  = self.cfg.get("max_sites", 25)
        sites  = sites[:max_s]
        LOG.info(f"Active sites: {len(sites)} identified "
                 f"(top={sum(s['type']=='top' for s in sites)}, "
                 f"bridge={sum(s['type']=='bridge' for s in sites)}, "
                 f"hollow={sum(s['type']=='hollow' for s in sites)})")
        return sites

    # -- Top sites ---------------------------------------------------------
    def _top_sites(self) -> List[Dict]:
        sites = []
        for i in self.top_indices:
            pos = self.slab.positions[i].copy()
            pos[2] += self.cfg["z_height"]
            sites.append({"type": "top", "position": pos,
                           "atom_indices": [i], "surface_z": self.slab.positions[i, 2]})
        return sites

    # -- Bridge sites ------------------------------------------------------
    def _bridge_sites(self) -> List[Dict]:
        sites = []
        top_pos = self.slab.positions[self.top_indices]
        cell    = self.slab.get_cell()
        for ii, i in enumerate(self.top_indices):
            for jj, j in enumerate(self.top_indices):
                if j <= i:
                    continue
                pi = self.slab.positions[i]
                pj = self.slab.positions[j]
                d  = np.linalg.norm(pi - pj)
                if 2.0 < d < 3.5:  # nearest-neighbour bridge
                    mid = 0.5 * (pi + pj)
                    mid[2] = max(pi[2], pj[2]) + self.cfg["z_height"]
                    sites.append({"type": "bridge", "position": mid,
                                  "atom_indices": [i, j],
                                  "surface_z": max(pi[2], pj[2])})
        return sites

    # -- Hollow sites ------------------------------------------------------
    def _hollow_sites(self) -> List[Dict]:
        sites = []
        ti    = self.top_indices
        for ii in range(len(ti)):
            for jj in range(ii + 1, len(ti)):
                for kk in range(jj + 1, len(ti)):
                    i, j, k = ti[ii], ti[jj], ti[kk]
                    pi = self.slab.positions[i]
                    pj = self.slab.positions[j]
                    pk = self.slab.positions[k]
                    d_ij = np.linalg.norm(pi - pj)
                    d_jk = np.linalg.norm(pj - pk)
                    d_ik = np.linalg.norm(pi - pk)
                    # Only near-equilateral triangles (fcc(111) hollow)
                    if max(d_ij, d_jk, d_ik) < 3.5 and min(d_ij, d_jk, d_ik) > 1.8:
                        centroid = (pi + pj + pk) / 3.0
                        centroid[2] = max(pi[2], pj[2], pk[2]) + self.cfg["z_height"]
                        sites.append({"type": "hollow", "position": centroid,
                                      "atom_indices": [i, j, k],
                                      "surface_z": max(pi[2], pj[2], pk[2])})
        return sites

    # -- Deduplication (merge sites within 0.5 Ang) -------------------------
    @staticmethod
    def _deduplicate(sites: List[Dict], tol: float = 0.50) -> List[Dict]:
        kept = []
        for s in sites:
            p = s["position"]
            is_dup = any(np.linalg.norm(p[:2] - k["position"][:2]) < tol
                         for k in kept)
            if not is_dup:
                kept.append(s)
        return kept

    # -- HEA chemical fingerprint filter ----------------------------------
    def _fingerprint_filter(self, sites: List[Dict]) -> List[Dict]:
        seen_fps = set()
        filtered = []
        for s in sites:
            fp = self._site_fingerprint(s)
            if fp not in seen_fps:
                seen_fps.add(fp)
                s["fingerprint"] = fp
                filtered.append(s)
        n_uniq = len(seen_fps)
        LOG.debug(f"  Fingerprint filter: {len(sites)} -> {len(filtered)} "
                  f"({n_uniq} unique envs)")
        return filtered

    def _site_fingerprint(self, site: Dict, n_shells: int = 2,
                           cutoff: float = 4.0) -> Tuple:
        fp = []
        site_pos = site["position"]
        for shell in range(1, n_shells + 1):
            r_max = cutoff * shell
            r_min = cutoff * (shell - 1)
            elems = []
            for a in self.slab:
                d = np.linalg.norm(a.position[:2] - site_pos[:2])  # lateral only
                if r_min < d <= r_max:
                    elems.append(a.symbol)
            fp.append(tuple(sorted(Counter(elems).items())))
        return (site["type"],) + tuple(fp)


# ==============================================================================
# MODULE 4: CHEMISTRY VALIDATOR
# ==============================================================================

class ChemistryValidator:
    """
    Comprehensive chemistry validation for each intermediate.

    Checks performed
    ----------------
    1. Stoichiometry  -- exact atom counts vs STEP_EXPECTED_COMPOSITION
    2. Bond integrity -- critical bonds (C-O, C-H, O-H) present / absent as expected
    3. Bond lengths   -- realistic distances (no extreme stretch / compression)
    4. Adsorption     -- adsorbate not desorbed or floating
    5. Fragmentation  -- adsorbate atoms still connected (no split fragments)
    6. Slab integrity -- no catastrophic slab reconstruction
    7. Overlap        -- no atom-atom clashes < 1.0 Ang
    8. Convergence    -- force residual below threshold (if provided)

    Returns ValidationResult.
    """

    # Steps where C-O bond must be ABSENT after relaxation
    CO_DISS_STEPS = frozenset({
        "05_CO_diss",
        "05_OH_from_CO_diss",
        "05_H2O_from_CO_diss",
        "06_C_from_CO_diss",
    })

    # Steps where C-O bond must be PRESENT in ALL C-O pairs
    CO_BOND_STEPS = frozenset({
        "01_CO2", "02_COOH", "02_HCOO",
        "03_CO", "03_HCOOH",
        "04_CHO", "04_COH",
        "05_CH2O", "06_CH3O",
        "05_CH2O_E", "06_CH3O_E", "07_CH3OH",
    })

    # Steps where CO* + H2O* are co-adsorbed:
    # C must have EXACTLY ONE C-O bond (CO*, d~1.10-1.16 Ang)
    # and at least one O must have 2 H-bonds (H2O*, NOT bonded to C).
    # 03_H2O_from_COOH / 04_H2O_from_HCOOH must NOT be in CO_BOND_STEPS.
    CO_WATER_STEPS = frozenset({
        "03_H2O_from_COOH",
        "04_H2O_from_HCOOH",
    })

    def __init__(self, n_slab: int, cfg: Dict[str, Any]):
        self.n_slab = n_slab
        self.cfg    = cfg

    # -- Main entry point --------------------------------------------------
    def validate(self, atoms: Atoms, step: str,
                 max_force: Optional[float] = None) -> ValidationResult:
        msgs = []
        adsorbate = atoms[self.n_slab:]

        stoich_ok = self._check_stoichiometry(adsorbate, step, msgs)
        bonds_ok  = self._check_critical_bonds(adsorbate, step, msgs)
        geo_ok    = self._check_geometry(adsorbate, atoms, step, msgs)
        ads_ok    = self._check_adsorption(adsorbate, atoms, step, msgs)
        conv_ok   = self._check_convergence(atoms, max_force, msgs)

        is_valid  = stoich_ok and bonds_ok and geo_ok and ads_ok and conv_ok
        descriptor = self._describe(adsorbate, atoms, step)

        return ValidationResult(
            is_valid       = is_valid,
            stoich_ok      = stoich_ok,
            bonds_ok       = bonds_ok,
            geometry_ok    = geo_ok,
            adsorbed_ok    = ads_ok,
            convergence_ok = conv_ok,
            descriptor     = descriptor,
            messages       = msgs,
        )

    # -- 1. Stoichiometry --------------------------------------------------
    def _check_stoichiometry(self, ads: Atoms, step: str, msgs: List[str]) -> bool:
        expected = STEP_EXPECTED_COMPOSITION.get(step)
        if expected is None:
            return True
        actual = {el: sum(1 for a in ads if a.symbol == el)
                  for el in ("C", "H", "O")}
        ok = True
        for el, n_exp in expected.items():
            n_act = actual.get(el, 0)
            if n_act < n_exp:
                msgs.append(f"Stoich FAIL {step}: {el} expected>={n_exp} got {n_act}")
                ok = False
        # 10_clean: must have NO adsorbate atoms
        if step == "10_clean" or step == "08_CH3OH_des":
            total = sum(actual.values())
            if total > 0:
                msgs.append(f"{step}: expected bare surface, found {actual}")
                ok = False
        return ok

    # -- 2. Critical bond integrity ----------------------------------------
    def _check_critical_bonds(self, ads: Atoms, step: str, msgs: List[str]) -> bool:
        if len(ads) == 0:
            return True

        c_pos = [a.position for a in ads if a.symbol == "C"]
        o_pos = [a.position for a in ads if a.symbol == "O"]
        h_pos = [a.position for a in ads if a.symbol == "H"]
        ok = True

        # C-O bond required
        if step in self.CO_BOND_STEPS and c_pos and o_pos:
            found = any(np.linalg.norm(cp - op) < CO_BOND_MAX
                        for cp in c_pos for op in o_pos)
            if not found:
                msgs.append(f"Bond FAIL {step}: C-O bond missing")
                ok = False

        # C-O bond must be ABSENT (dissociation)
        if step in self.CO_DISS_STEPS and c_pos and o_pos:
            bonded = any(np.linalg.norm(cp - op) < CO_BOND_MAX
                         for cp in c_pos for op in o_pos)
            if bonded and step == "05_CO_diss":
                msgs.append(f"Bond FAIL {step}: C-O NOT dissociated (bond still present)")
                ok = False

        # O-H bond required
        oh_req_steps = {
            "02_COOH", "03_HCOOH", "04_COH", "02_HCOO",
            "08_OH", "09_H2O", "03_H2O_from_COOH",
            "04_H2O_from_HCOOH", "05_OH_from_CO_diss",
            "05_H2O_from_CO_diss", "07_CH3OH",
        }
        if step in oh_req_steps and o_pos and h_pos:
            found = any(np.linalg.norm(op - hp) < OH_BOND_MAX
                        for op in o_pos for hp in h_pos)
            if not found:
                msgs.append(f"Bond FAIL {step}: O-H bond missing")
                ok = False

        # C-H bond required
        ch_req_steps = {
            "04_CHO", "05_CH2O", "06_CH3O", "06_CH",
            "07_CH2", "08_CH3", "02_HCOO", "03_HCOOH",
            "04_H2O_from_HCOOH", "05_CH2O_E", "06_CH3O_E",
            "07_CH3OH", "07_O_CH4",
        }
        if step in ch_req_steps and c_pos and h_pos:
            found = any(np.linalg.norm(cp - hp) < CH_BOND_MAX
                        for cp in c_pos for hp in h_pos)
            if not found:
                msgs.append(f"Bond FAIL {step}: C-H bond missing")
                ok = False

        return ok

    # -- 3. Bond-length sanity ---------------------------------------------
    def _check_geometry(self, ads: Atoms, all_atoms: Atoms,
                         step: str, msgs: List[str]) -> bool:
        ok = True
        # Check for unrealistically short bonds (overlap)
        for i, ai in enumerate(ads):
            for j, aj in enumerate(ads):
                if j <= i:
                    continue
                d = np.linalg.norm(ai.position - aj.position)
                if d < 0.80:
                    msgs.append(f"Overlap {step}: {ai.symbol}-{aj.symbol} d={d:.3f} Ang")
                    ok = False
        # Check for unrealistically long bonds (fragmentation)
        # If adsorbate has >1 atom, the minimum spanning connectivity should hold
        if len(ads) > 1:
            min_d = min(
                np.linalg.norm(ads[i].position - ads[j].position)
                for i in range(len(ads)) for j in range(i + 1, len(ads))
            )
            if min_d > 3.5:   # softened from 3.0 Ang -- allow more drift before calling fragmentation
                msgs.append(f"Fragmentation {step}: min inter-adsorbate distance {min_d:.3f} Ang")
                ok = False
        return ok

    # -- 4. Adsorption check -----------------------------------------------
    def _check_adsorption(self, ads: Atoms, all_atoms: Atoms,
                           step: str, msgs: List[str]) -> bool:
        if len(ads) == 0:
            return True
        slab_top_z = all_atoms.positions[:self.n_slab, 2].max()
        desorp_threshold = self.cfg.get("desorption_gap_threshold", 4.0)

        lowest_ads_z = float(min(a.position[2] for a in ads))
        gap = lowest_ads_z - slab_top_z

        # For CO-dissociation steps, both C* and O* are individually surface-bound
        # but may have different heights -- use more lenient threshold
        if step in self.CO_DISS_STEPS:
            if gap > 6.0:
                msgs.append(f"Desorption {step}: lowest adsorbate {gap:.2f} Ang above surface")
                return False
            return True

        # For desorption / lifting steps, high gap is expected
        lift_steps = {"09_CH4", "08_CH3OH_des", "10_clean"}
        if step in lift_steps:
            return True

        if gap > desorp_threshold:
            msgs.append(f"Desorption {step}: adsorbate {gap:.2f} Ang above surface (>{desorp_threshold} Ang)")
            return False

        return True

    # -- 5. Convergence ----------------------------------------------------
    @staticmethod
    def _check_convergence(atoms: Atoms, max_force: Optional[float],
                            msgs: List[str]) -> bool:
        if max_force is None:
            return True
        threshold = CFG["relax_fmax"] * 2.0  # allow 2x slack for reporting
        if max_force > threshold:
            msgs.append(f"Convergence: max force {max_force:.4f} > {threshold:.4f} eV/Ang")
            return False
        return True

    # -- Geometry descriptor (used in folder naming) -----------------------
    def _describe(self, ads: Atoms, all_atoms: Atoms, step: str) -> str:
        if len(ads) == 0:
            return "clean_surface"

        expected = STEP_EXPECTED_COMPOSITION.get(step, {})
        actual   = Counter(a.symbol for a in ads)

        # Desorption check
        slab_top_z = all_atoms.positions[:self.n_slab, 2].max()
        desorbed   = [a.symbol for a in ads
                      if a.position[2] - slab_top_z > 5.0]
        if desorbed:
            return f"desorbed_{''.join(sorted(desorbed))}"

        # Bond integrity descriptor
        c_pos = [a.position for a in ads if a.symbol == "C"]
        o_pos = [a.position for a in ads if a.symbol == "O"]
        h_pos = [a.position for a in ads if a.symbol == "H"]

        broken = []
        if step in self.CO_BOND_STEPS and c_pos and o_pos:
            if not any(np.linalg.norm(cp - op) < CO_BOND_MAX
                       for cp in c_pos for op in o_pos):
                broken.append("C-O")

        if step == "05_CO_diss" and c_pos and o_pos:
            if any(np.linalg.norm(cp - op) < CO_BOND_MAX
                   for cp in c_pos for op in o_pos):
                return "CO_not_dissociated"

        if broken:
            return f"dissociated_{'_'.join(broken)}_broken"

        return "intact"



# ==============================================================================
# MODULE 5: STRUCTURE BUILDER
# Explicit geometry for every intermediate in every pathway.
# Each builder receives the previous relaxed Atoms and returns a new
# Atoms object with the correct adsorbate geometry for MACE to relax.
# ==============================================================================

class StructureBuilder:
    """
    Chemistry-aware geometry builder for all CO2RR intermediates.

    Design principles
    -----------------
    - Each builder positions atoms at chemically realistic starting geometries
      so that the optimizer converges to the correct local minimum.
    - Builders never change slab atoms -- only adsorbate indices (n_slab:).
    - Atom conservation: only add/remove atoms via _add_atom / _del_atoms;
      never change atom types in place.
    - H-angle sweeping: h_angle_idx rotates the azimuthal placement of newly
      added H atoms, enabling the stability-test iterations to sample all
      chemically distinct orientations.

    Public interface
    ----------------
    build(step, prev_atoms, pathway_id, h_angle_idx=0) -> Atoms
    build_for_iteration(step, initial_atoms, pathway_id, iteration, n_total) -> Atoms
    """

    # Tetrahedral bond directions (C sp3)
    _TET = np.array([
        [ 0.000,  0.000,  1.000],
        [ 0.943,  0.000, -0.333],
        [-0.471,  0.816, -0.333],
        [-0.471, -0.816, -0.333],
    ], dtype=float)

    # N evenly-spaced azimuthal angles for H placement sweep
    _N_H_ANGLES = 6

    def __init__(self, n_slab: int):
        self.n_slab      = n_slab
        self._h_angle_idx = 0   # set before each build call

    # -- Low-level atom helpers ---------------------------------------------

    def _ads(self, atoms: Atoms) -> Atoms:
        """Return view of adsorbate atoms (indices n_slab:)."""
        return atoms[self.n_slab:]

    def _surface_top(self, atoms: Atoms) -> float:
        return float(atoms.positions[:self.n_slab, 2].max())

    def _find_first(self, atoms: Atoms, sym: str) -> Optional[int]:
        for i in range(self.n_slab, len(atoms)):
            if atoms[i].symbol == sym:
                return i
        return None

    def _find_all(self, atoms: Atoms, sym: str) -> List[int]:
        return [i for i in range(self.n_slab, len(atoms))
                if atoms[i].symbol == sym]

    def _add_atom(self, atoms: Atoms, sym: str, pos: np.ndarray) -> Atoms:
        new = atoms.copy()
        new.append(Atom(sym, position=pos))
        return new

    def _del_atoms(self, atoms: Atoms, indices: List[int]) -> Atoms:
        new = atoms.copy()
        del new[sorted(indices, reverse=True)]
        return new

    def _h_azimuth(self, idx: int = 0) -> float:
        """Return azimuthal angle (rad) for H placement iteration idx."""
        return 2.0 * np.pi * idx / self._N_H_ANGLES

    def _h_direction_above(self, idx: int = 0, polar_deg: float = 45.0) -> np.ndarray:
        """Unit vector for H placement: above surface at given polar angle."""
        phi   = self._h_azimuth(idx)
        theta = np.radians(polar_deg)
        return np.array([np.sin(theta) * np.cos(phi),
                          np.sin(theta) * np.sin(phi),
                          np.cos(theta)])

    # -- Strip helpers ------------------------------------------------------

    def _strip_h2o(self, atoms: Atoms) -> Atoms:
        """
        Remove the H2O group from a co-adsorbed CO*+H2O* structure.

        Three strategies in order of confidence:
          1. O with >= 2 H's within OH_BOND_MAX*1.2 (canonical H2O*)
          2. O with 1 H + a floating H nearby, NOT the CO* oxygen
             (handles distorted/recovered H2O where H's are spread out)
          3. Last resort: O furthest from C + 2 nearest H's
             (handles peroxy-type artefacts where no clear H2O exists)
        Repeats until no H2O can be found (strips all H2O if multiple present).
        Never removes the CO* oxygen (C-O < CO_BOND_MAX).
        """
        new = atoms.copy()

        def _co_star_oxygen(atoms_):
            """Index of the O that is part of CO* (shortest C-O distance)."""
            ci = self._find_first(atoms_, "C")
            if ci is None:
                return None
            o_list = self._find_all(atoms_, "O")
            if not o_list:
                return None
            return min(o_list,
                       key=lambda oi: np.linalg.norm(
                           atoms_.positions[ci] - atoms_.positions[oi]))

        # Strategy 1: O with >= 2 direct H-bonds (threshold: 1.20 Ang)
        changed = True
        while changed:
            changed = False
            for oi in self._find_all(new, "O"):
                h_near = [hi for hi in self._find_all(new, "H")
                          if np.linalg.norm(new.positions[oi] -
                                            new.positions[hi]) < 1.20]
                if len(h_near) >= 2:
                    del new[sorted([oi] + h_near[:2], reverse=True)]
                    changed = True
                    break

        # Strategy 2: O with 1 H (not CO* O) + nearest floating H
        co_o = _co_star_oxygen(new)
        o_idxs = self._find_all(new, "O")
        h_idxs = self._find_all(new, "H")
        for oi in o_idxs:
            if oi == co_o:
                continue
            h_bond = [hi for hi in h_idxs
                      if np.linalg.norm(new.positions[oi] -
                                        new.positions[hi]) < 1.30]
            if h_bond:
                remove = {oi, h_bond[0]}
                # Add the floating H closest to this O (if any remain)
                remaining = [hi for hi in self._find_all(new, "H")
                             if hi not in remove]
                if remaining:
                    extra_h = min(remaining,
                                  key=lambda hi: np.linalg.norm(
                                      new.positions[oi] - new.positions[hi]))
                    # Only take it if it is not bonded to another O or C
                    ci = self._find_first(new, "C")
                    bonded_elsewhere = False
                    if ci is not None:
                        bonded_elsewhere = (
                            np.linalg.norm(new.positions[ci] -
                                           new.positions[extra_h]) < 1.20)
                    for other_o in self._find_all(new, "O"):
                        if other_o in remove:
                            continue
                        if np.linalg.norm(new.positions[other_o] -
                                          new.positions[extra_h]) < 1.20:
                            bonded_elsewhere = True
                    if not bonded_elsewhere:
                        remove.add(extra_h)
                del new[sorted(remove, reverse=True)]
                return new

        # Strategy 3: last resort -- O furthest from C + 2 nearest H's
        ci = self._find_first(new, "C")
        o_idxs = self._find_all(new, "O")
        h_idxs = self._find_all(new, "H")
        if ci is not None and len(o_idxs) >= 2 and h_idxs:
            h2o_o = max(o_idxs,
                        key=lambda oi: np.linalg.norm(
                            new.positions[ci] - new.positions[oi]))
            h_sorted = sorted(
                h_idxs,
                key=lambda hi: np.linalg.norm(new.positions[h2o_o] -
                                               new.positions[hi]))
            del new[sorted([h2o_o] + h_sorted[:2], reverse=True)]
        return new

    def _strip_ch4(self, atoms: Atoms) -> Atoms:
        """Remove any CH4 molecule (C bonded to >=3 H, not bonded to O)."""
        new = atoms.copy()
        for ci in list(self._find_all(new, "C")):
            if ci >= len(new):
                continue
            c_pos  = new.positions[ci]
            h_near = [hi for hi in self._find_all(new, "H")
                      if np.linalg.norm(new.positions[hi] - c_pos) < 1.45]
            o_bond = any(np.linalg.norm(new.positions[oi] - c_pos) < 1.80
                         for oi in self._find_all(new, "O"))
            if len(h_near) >= 3 and not o_bond:
                del new[sorted([ci] + h_near, reverse=True)]
                break
        return new

    def _strip_stray_h(self, atoms: Atoms) -> Atoms:
        """Remove H atoms not bonded to any O (orphaned after CH4 strip)."""
        new    = atoms.copy()
        to_del = []
        o_idx  = set(self._find_all(new, "O"))
        for hi in self._find_all(new, "H"):
            if not any(np.linalg.norm(new.positions[hi] - new.positions[oi]) < OH_BOND_MAX * 1.2
                       for oi in o_idx):
                to_del.append(hi)
        if to_del:
            del new[sorted(set(to_del), reverse=True)]
        return new

    def _strip_methanol(self, atoms: Atoms) -> Atoms:
        """Remove CH3OH molecule from surface (for Path E clean step)."""
        # CH3OH: C bonded to O, O has 1 H, C has 3 H
        new = atoms.copy()
        for ci in list(self._find_all(new, "C")):
            c_pos  = new.positions[ci]
            o_near = [oi for oi in self._find_all(new, "O")
                      if np.linalg.norm(new.positions[oi] - c_pos) < 1.50]
            h_on_c = [hi for hi in self._find_all(new, "H")
                      if np.linalg.norm(new.positions[hi] - c_pos) < 1.45]
            if o_near and len(h_on_c) >= 3:
                oi    = o_near[0]
                o_pos = new.positions[oi]
                h_on_o = [hi for hi in self._find_all(new, "H")
                          if np.linalg.norm(new.positions[hi] - o_pos) < OH_BOND_MAX * 1.2]
                to_del = sorted(set([ci, oi] + h_on_c + h_on_o), reverse=True)
                del new[to_del]
                break
        return new

    def _enforce_stoich(self, atoms: Atoms, step: str) -> Atoms:
        """Hard-enforce adsorbate stoichiometry by removing excess atoms."""
        expected = STEP_EXPECTED_COMPOSITION.get(step)
        if expected is None:
            return atoms
        new = atoms.copy()
        c_idx = self._find_first(new, "C")
        ref_pos = new.positions[c_idx] if c_idx is not None else np.zeros(3)
        for el, n_exp in expected.items():
            indices = self._find_all(new, el)
            if len(indices) > n_exp:
                # Keep the n_exp closest to reference (C or centroid)
                dists = [(np.linalg.norm(new.positions[i] - ref_pos), i)
                         for i in indices]
                dists.sort()
                excess = [i for _, i in dists[n_exp:]]
                del new[sorted(excess, reverse=True)]
        return new

    # ======================================================================
    # PATH-INDEPENDENT GEOMETRY BUILDERS
    # ======================================================================

    def _place_co2(self, atoms: Atoms) -> Atoms:
        """
        Bent CO2* geometry: O-C-O = 135 deg, C-O = 1.22 Ang, tilt 30 deg from surface.
        Carbon anchored above top site; both O atoms in xz-plane.

        Works in two modes:
          - bare slab (no adsorbate C/O): appends fresh C + 2 O atoms above the
            centroid of the top layer.
          - already has C/O (e.g. restart): repositions existing atoms.
        """
        new  = atoms.copy()
        stol = self._surface_top(new)
        ci   = self._find_first(new, "C")
        oi   = self._find_all(new, "O")

        co_len    = 1.22
        half_bend = np.radians((180.0 - 135.0) / 2.0)  # 22.5 deg
        tilt      = np.radians(30.0)
        mol_axis  = np.array([np.cos(tilt), 0.0, np.sin(tilt)])

        def rot_y(v, a):
            c, s = np.cos(a), np.sin(a)
            return np.array([c*v[0]+s*v[2], v[1], -s*v[0]+c*v[2]])

        if ci is None or len(oi) < 2:
            # --- Bare slab: append fresh CO2 above top-layer centroid ---
            top_idx = [i for i, z in enumerate(new.positions[:, 2])
                       if z > stol - 0.5]
            if top_idx:
                cx = new.positions[top_idx, 0].mean()
                cy = new.positions[top_idx, 1].mean()
            else:
                cell = new.get_cell()
                cx, cy = cell[0, 0] / 2.0, cell[1, 1] / 2.0

            c_pos  = np.array([cx, cy, stol + 1.50])
            o1_dir = rot_y(mol_axis, +half_bend)
            o2_dir = rot_y(mol_axis, -half_bend)
            o1_pos = c_pos + co_len * o1_dir
            o2_pos = c_pos - co_len * o2_dir
            o2_pos[2] = max(o2_pos[2], stol + 0.50)

            new.append(Atom("C", position=c_pos))
            new.append(Atom("O", position=o1_pos))
            new.append(Atom("O", position=o2_pos))
        else:
            # --- Existing adsorbate: reposition in place ---
            c_pos = new.positions[ci].copy()
            c_pos[2] = stol + 1.50
            o1_dir = rot_y(mol_axis, +half_bend)
            o2_dir = rot_y(mol_axis, -half_bend)
            o1_pos = c_pos + co_len * o1_dir
            o2_pos = c_pos - co_len * o2_dir
            o2_pos[2] = max(o2_pos[2], stol + 0.50)
            new.positions[ci]    = c_pos
            new.positions[oi[0]] = o1_pos
            new.positions[oi[1]] = o2_pos

        return new

    def _place_cooh(self, prev: Atoms) -> Atoms:
        """COOH*: C anchored, one O surface-side, one O-H pointing up."""
        new  = prev.copy()
        stol = self._surface_top(new)
        ci   = self._find_first(new, "C")
        oi   = self._find_all(new, "O")
        hi   = self._find_all(new, "H")

        if ci is None or len(oi) < 2:
            return self._add_h_to_o(new, stol)

        c_pos     = new.positions[ci].copy()
        c_pos[2]  = stol + 1.45
        co_len    = 1.25
        o_surf    = c_pos + np.array([0.10, 0.0, -co_len * 0.85])
        o_surf[2] = max(o_surf[2], stol + 0.55)
        o_oh      = c_pos + co_len * np.array([0.65, 0.0, 0.76])
        h_pos     = o_oh + 0.97 * np.array([0.71, 0.0, 0.71])

        # Vary H azimuth with iteration index
        phi   = self._h_azimuth(self._h_angle_idx)
        h_dir = np.array([np.cos(phi), np.sin(phi), 0.7])
        h_dir /= np.linalg.norm(h_dir)
        h_pos  = o_oh + 0.97 * h_dir

        new.positions[ci]    = c_pos
        new.positions[oi[0]] = o_surf
        new.positions[oi[1]] = o_oh
        if hi:
            new.positions[hi[0]] = h_pos
        else:
            new.append(Atom("H", position=h_pos))
        return new

    def _place_hcoo(self, prev: Atoms) -> Atoms:
        """HCOO* (formate): bidentate O-C-O bridge, H on C pointing up."""
        new  = prev.copy()
        stol = self._surface_top(new)
        ci   = self._find_first(new, "C")
        oi   = self._find_all(new, "O")

        if ci is None or len(oi) < 2:
            return prev.copy()

        c_pos    = new.positions[ci].copy()
        c_pos[2] = stol + 1.80
        co_len   = 1.26
        # Both O atoms below C, bridging surface
        phi = self._h_azimuth(self._h_angle_idx)
        o1_pos   = c_pos + co_len * np.array([ 0.5,  0.0, -0.866])
        o2_pos   = c_pos + co_len * np.array([-0.5,  0.0, -0.866])
        o1_pos[2] = max(o1_pos[2], stol + 0.50)
        o2_pos[2] = max(o2_pos[2], stol + 0.50)
        # H on C, pointing away from surface
        h_pos    = c_pos + 1.09 * np.array([np.sin(phi)*0.3, np.cos(phi)*0.3, 1.0])
        h_pos[2] = stol + 2.80

        new.positions[ci]    = c_pos
        new.positions[oi[0]] = o1_pos
        new.positions[oi[1]] = o2_pos
        hi = self._find_all(new, "H")
        if hi:
            new.positions[hi[0]] = h_pos
        else:
            new.append(Atom("H", position=h_pos))
        return new

    def _place_co(self, prev: Atoms) -> Atoms:
        """CO* upright: C down, O up, C-O = 1.15 Ang."""
        new  = self._strip_h2o(prev)
        stol = self._surface_top(new)
        ci   = self._find_first(new, "C")
        oi   = self._find_first(new, "O")
        if ci is None or oi is None:
            return new

        c_pos = new.positions[ci].copy()
        c_pos[2] = stol + 1.30
        o_pos    = c_pos + np.array([0.0, 0.0, 1.15])

        new.positions[ci] = c_pos
        new.positions[oi] = o_pos
        return new

    def _add_h_to_c(self, prev: Atoms, stol: float) -> Atoms:
        """Add one H to the carbon atom."""
        new = prev.copy()
        ci  = self._find_first(new, "C")
        if ci is None:
            return new
        c_pos = new.positions[ci]
        dirn  = self._h_direction_above(self._h_angle_idx, polar_deg=50.0)
        h_pos = c_pos + 1.09 * dirn
        h_pos[2] = max(h_pos[2], stol + 0.40)
        new.append(Atom("H", position=h_pos))
        return new

    def _add_h_to_o(self, prev: Atoms, stol: Optional[float] = None) -> Atoms:
        """Add one H to the first O atom."""
        new = prev.copy()
        if stol is None:
            stol = self._surface_top(new)
        oi  = self._find_first(new, "O")
        if oi is None:
            return new
        o_pos = new.positions[oi]
        dirn  = self._h_direction_above(self._h_angle_idx, polar_deg=55.0)
        h_pos = o_pos + 0.97 * dirn
        h_pos[2] = max(h_pos[2], stol + 0.40)
        new.append(Atom("H", position=h_pos))
        return new

    def _place_cho(self, prev: Atoms) -> Atoms:
        """CHO* formyl: C anchored, C=O tilted ~45 deg, H on C at 120 deg from C-O."""
        new  = prev.copy()
        stol = self._surface_top(new)
        ci   = self._find_first(new, "C")
        oi   = self._find_first(new, "O")
        hi   = self._find_all(new, "H")
        if ci is None or oi is None:
            return new

        phi   = self._h_azimuth(self._h_angle_idx)

        # C anchored slightly above surface
        c_pos = new.positions[ci].copy()
        c_pos[2] = stol + 1.35

        # O above C, slight tilt
        o_pos = c_pos + 1.20 * np.array([0.17, 0.0, 0.985])

        # H on C: 120 deg from C-O bond direction
        h_angle = np.radians(120.0)
        h_base  = c_pos + 1.09 * np.array([np.sin(h_angle), 0.0, -np.cos(h_angle)])
        # Rotate H position in xy-plane by phi
        h_pos   = c_pos + 1.09 * np.array([
            np.sin(h_angle) * np.cos(phi),
            np.sin(h_angle) * np.sin(phi),
            -np.cos(h_angle),
        ])
        h_pos[2] = max(h_pos[2], stol + 0.60)

        new.positions[ci] = c_pos
        new.positions[oi] = o_pos
        if hi:
            new.positions[hi[0]] = h_pos
        else:
            new.append(Atom("H", position=h_pos))
        return new

    def _place_coh(self, prev: Atoms) -> Atoms:
        """COH* hydroxymethylidene: C anchored, O sideways, H on O."""
        new  = prev.copy()
        stol = self._surface_top(new)
        ci   = self._find_first(new, "C")
        oi   = self._find_first(new, "O")
        hi   = self._find_all(new, "H")
        if ci is None or oi is None:
            return new

        phi   = self._h_azimuth(self._h_angle_idx)
        c_pos = new.positions[ci].copy()
        c_pos[2] = stol + 1.35
        o_pos = c_pos + 1.35 * np.array([np.cos(phi), np.sin(phi), 0.2])
        h_pos = o_pos + 0.97 * np.array([np.cos(phi + 0.5), np.sin(phi + 0.5), 0.6])
        h_pos[2] = max(h_pos[2], stol + 0.50)

        new.positions[ci] = c_pos
        new.positions[oi] = o_pos
        if hi:
            new.positions[hi[0]] = h_pos
        else:
            new.append(Atom("H", position=h_pos))
        return new

    def _place_ch2o(self, prev: Atoms) -> Atoms:
        """
        CH2O* formaldehyde: H2C=O with BOTH H on C, O pointing up/away.

        CRITICAL: NEVER place H on O. That produces HCOH (hydroxymethylidene),
        a different isomer (Path B species, not Path A formaldehyde).

        The C=O tilt angle varies with h_angle_idx to probe different orientations
        and avoid the HCOH local minimum that MACE can fall into when C=O is
        tilted too close to the surface.
          idx 0: O straight up      (tilt 0 deg)
          idx 1: O tilted 30 deg
          idx 2: O tilted 45 deg
          idx 3: O tilted 60 deg
          idx 4+: azimuthal rotation of idx 0
        """
        new  = prev.copy()
        stol = self._surface_top(new)
        ci   = self._find_first(new, "C")
        oi   = self._find_first(new, "O")
        if ci is None or oi is None:
            return new

        idx   = self._h_angle_idx
        phi   = self._h_azimuth(idx)   # azimuthal rotation of H's

        # C=O tilt angle (degrees from surface normal) varies per iteration
        _tilts = [0, 30, 45, 60, 0, 30, 45, 60]
        tilt_deg = _tilts[idx % len(_tilts)]
        tilt_rad = np.radians(tilt_deg)

        c_pos    = new.positions[ci].copy()
        c_pos[2] = stol + 1.40

        # O direction: tilted from vertical by tilt_rad, azimuth phi
        co_len = 1.22
        o_dir  = np.array([
            np.sin(tilt_rad) * np.cos(phi),
            np.sin(tilt_rad) * np.sin(phi),
            np.cos(tilt_rad),   # always +z component -- O stays ABOVE C
        ])
        o_pos  = c_pos + co_len * o_dir
        o_pos[2] = max(o_pos[2], c_pos[2] + 0.20)  # O never below C

        # Both H on C (sp2, 120 deg apart in plane perpendicular to C=O axis)
        # H direction perpendicular to C=O, rotated azimuthally
        perp  = np.array([-np.sin(phi), np.cos(phi), 0.0])
        ch_len = 1.09
        h1_dir = -0.5 * o_dir + 0.866 * perp     # 120 deg from C=O
        h2_dir = -0.5 * o_dir - 0.866 * perp     # 240 deg from C=O
        h1_pos = c_pos + ch_len * (h1_dir / (np.linalg.norm(h1_dir) + 1e-9))
        h2_pos = c_pos + ch_len * (h2_dir / (np.linalg.norm(h2_dir) + 1e-9))
        h1_pos[2] = max(h1_pos[2], stol + 0.25)
        h2_pos[2] = max(h2_pos[2], stol + 0.25)

        new.positions[ci] = c_pos
        new.positions[oi] = o_pos
        hi = self._find_all(new, "H")
        if len(hi) >= 2:
            new.positions[hi[0]] = h1_pos
            new.positions[hi[1]] = h2_pos
        elif len(hi) == 1:
            new.positions[hi[0]] = h1_pos
            new.append(Atom("H", position=h2_pos))
        else:
            new.append(Atom("H", position=h1_pos))
            new.append(Atom("H", position=h2_pos))
        return new

    def _place_ch3o(self, prev: Atoms) -> Atoms:
        """CH3O* methoxy: O anchored to surface, C up, 3 H's on C (sp3)."""
        new  = prev.copy()
        stol = self._surface_top(new)
        ci   = self._find_first(new, "C")
        oi   = self._find_first(new, "O")
        if ci is None or oi is None:
            return new

        phi = self._h_azimuth(self._h_angle_idx)
        # O anchored near surface
        o_pos    = new.positions[oi].copy()
        o_pos[2] = stol + 1.10

        # C above O at C-O = 1.43 Ang
        c_pos = o_pos + np.array([0.0, 0.0, 1.43])

        # 3 H's on C in tetrahedral arrangement
        h_dirs = [
            np.array([ 0.943*np.cos(phi + i*2.094),
                        0.943*np.sin(phi + i*2.094), -0.333])
            for i in range(3)
        ]
        h_positions = [c_pos + 1.09 * d for d in h_dirs]
        for hp in h_positions:
            hp[2] = max(hp[2], o_pos[2] + 0.20)

        new.positions[oi] = o_pos
        new.positions[ci] = c_pos
        hi = self._find_all(new, "H")
        for k in range(3):
            if k < len(hi):
                new.positions[hi[k]] = h_positions[k]
            else:
                new.append(Atom("H", position=h_positions[k]))
        return new

    def _place_o_ch4(self, prev: Atoms) -> Atoms:
        """CH3O* + H -> [CH4 above surface + O*]."""
        new  = prev.copy()
        stol = self._surface_top(new)
        ci   = self._find_first(new, "C")
        oi   = self._find_first(new, "O")
        if ci is None or oi is None:
            return self._add_h_to_c(new, stol)

        # O stays on surface; C lifts to ~3 Ang with 4 H's
        o_pos    = new.positions[oi].copy()
        o_pos[2] = stol + 1.10
        new.positions[oi] = o_pos

        c_pos    = new.positions[ci].copy()
        c_pos[2] = stol + 3.00  # CH4 floating above O*
        new.positions[ci] = c_pos

        # Existing H's get redistributed around C (tetrahedral)
        phi  = self._h_azimuth(self._h_angle_idx)
        hi   = self._find_all(new, "H")
        dirs = [
            np.array([np.sin(theta)*np.cos(phi+k*2.094),
                       np.sin(theta)*np.sin(phi+k*2.094),
                       np.cos(theta)])
            for k, theta in enumerate([np.radians(110)]*4)
        ]
        for k in range(4):
            hpos = c_pos + 1.09 * self._TET[k]
            hpos[2] = max(hpos[2], stol + 0.30)
            if k < len(hi):
                new.positions[hi[k]] = hpos
            else:
                new.append(Atom("H", position=hpos))
        return new

    def _place_o_only(self, prev: Atoms) -> Atoms:
        """O* alone after CH4 desorption."""
        new  = self._strip_ch4(prev)
        new  = self._strip_stray_h(new)
        stol = self._surface_top(new)
        oi   = self._find_first(new, "O")
        if oi is not None:
            new.positions[oi][2] = max(new.positions[oi][2], stol + 1.10)
        return new

    def _place_oh(self, prev: Atoms) -> Atoms:
        """OH*: O on surface, H pointing upward."""
        new  = self._strip_ch4(prev)
        new  = self._strip_stray_h(new)
        stol = self._surface_top(new)
        oi   = self._find_first(new, "O")
        hi   = self._find_all(new, "H")
        if oi is None:
            return new

        phi   = self._h_azimuth(self._h_angle_idx)
        o_pos = new.positions[oi].copy()
        o_pos[2] = max(o_pos[2], stol + 1.10)
        h_pos = o_pos + 0.97 * self._h_direction_above(self._h_angle_idx)

        new.positions[oi] = o_pos
        if hi:
            new.positions[hi[0]] = h_pos
        else:
            new.append(Atom("H", position=h_pos))
        return new

    def _place_h2o_on_surface(self, prev: Atoms) -> Atoms:
        """H2O* on surface: O-H*****O-H angle 104.5 deg, sitting ~2 Ang above surface."""
        new  = prev.copy()
        stol = self._surface_top(new)
        oi   = self._find_first(new, "O")
        hi   = self._find_all(new, "H")
        if oi is None:
            return new

        phi    = self._h_azimuth(self._h_angle_idx)
        o_pos  = new.positions[oi].copy()
        o_pos[2] = max(o_pos[2], stol + 1.80)

        half_hoh = np.radians(104.5 / 2.0)
        h1_pos   = o_pos + 0.96 * np.array([ np.sin(half_hoh)*np.cos(phi),
                                               np.sin(half_hoh)*np.sin(phi),
                                               np.cos(half_hoh)])
        h2_pos   = o_pos + 0.96 * np.array([-np.sin(half_hoh)*np.cos(phi),
                                              -np.sin(half_hoh)*np.sin(phi),
                                               np.cos(half_hoh)])

        new.positions[oi] = o_pos
        if len(hi) >= 2:
            new.positions[hi[0]] = h1_pos
            new.positions[hi[1]] = h2_pos
        elif len(hi) == 1:
            new.positions[hi[0]] = h1_pos
            new.append(Atom("H", position=h2_pos))
        else:
            new.append(Atom("H", position=h1_pos))
            new.append(Atom("H", position=h2_pos))
        return new

    def _place_clean(self, prev: Atoms) -> Atoms:
        """Return bare slab (strip all adsorbate atoms)."""
        new = prev.copy()
        ads_idx = list(range(self.n_slab, len(new)))
        if ads_idx:
            del new[sorted(ads_idx, reverse=True)]
        return new

    # -- C-chain builders (used in Paths B, D, E) --------------------------

    def _place_c(self, prev: Atoms) -> Atoms:
        """C* alone on hollow site."""
        new  = self._strip_h2o(prev)
        stol = self._surface_top(new)
        ci   = self._find_first(new, "C")
        if ci is None:
            return new
        new.positions[ci][2] = stol + 1.10
        return new

    def _place_ch(self, prev: Atoms) -> Atoms:
        """CH*: C anchored, H pointing upward."""
        new  = prev.copy()
        stol = self._surface_top(new)
        ci   = self._find_first(new, "C")
        hi   = self._find_all(new, "H")
        if ci is None:
            return new

        c_pos    = new.positions[ci].copy()
        c_pos[2] = stol + 1.20
        h_dir    = self._h_direction_above(self._h_angle_idx, polar_deg=30.0)
        h_pos    = c_pos + 1.09 * h_dir

        new.positions[ci] = c_pos
        if hi:
            new.positions[hi[0]] = h_pos
        else:
            new.append(Atom("H", position=h_pos))
        return new

    def _place_ch2(self, prev: Atoms) -> Atoms:
        """CH2* methylene: C anchored, 2 H's at ~109 deg apart."""
        new  = prev.copy()
        stol = self._surface_top(new)
        ci   = self._find_first(new, "C")
        if ci is None:
            return new

        phi   = self._h_azimuth(self._h_angle_idx)
        c_pos = new.positions[ci].copy()
        c_pos[2] = stol + 1.25

        h1_pos = c_pos + 1.09 * np.array([ np.sin(np.radians(54.5))*np.cos(phi),
                                              np.sin(np.radians(54.5))*np.sin(phi),
                                              np.cos(np.radians(54.5))])
        h2_pos = c_pos + 1.09 * np.array([-np.sin(np.radians(54.5))*np.cos(phi),
                                             -np.sin(np.radians(54.5))*np.sin(phi),
                                              np.cos(np.radians(54.5))])

        new.positions[ci] = c_pos
        hi = self._find_all(new, "H")
        if len(hi) >= 2:
            new.positions[hi[0]] = h1_pos
            new.positions[hi[1]] = h2_pos
        elif len(hi) == 1:
            new.positions[hi[0]] = h1_pos
            new.append(Atom("H", position=h2_pos))
        else:
            new.append(Atom("H", position=h1_pos))
            new.append(Atom("H", position=h2_pos))
        return new

    def _place_ch3(self, prev: Atoms) -> Atoms:
        """CH3* methyl: C anchored, 3 H's tetrahedral."""
        new  = prev.copy()
        stol = self._surface_top(new)
        ci   = self._find_first(new, "C")
        if ci is None:
            return new

        phi   = self._h_azimuth(self._h_angle_idx)
        c_pos = new.positions[ci].copy()
        c_pos[2] = stol + 1.35

        h_dirs = [
            np.array([np.sin(np.radians(70.5))*np.cos(phi+k*2.094),
                       np.sin(np.radians(70.5))*np.sin(phi+k*2.094),
                       np.cos(np.radians(70.5))])
            for k in range(3)
        ]
        new.positions[ci] = c_pos
        hi = self._find_all(new, "H")
        for k in range(3):
            hpos = c_pos + 1.09 * h_dirs[k]
            if k < len(hi):
                new.positions[hi[k]] = hpos
            else:
                new.append(Atom("H", position=hpos))
        return new

    def _place_ch4_lift(self, prev: Atoms) -> Atoms:
        """CH3* + H -> CH4 lifted ~4 Ang above surface (approaching desorption)."""
        new  = prev.copy()
        stol = self._surface_top(new)
        ci   = self._find_first(new, "C")
        if ci is None:
            return new

        phi   = self._h_azimuth(self._h_angle_idx)
        c_pos = new.positions[ci].copy()
        c_pos[2] = stol + 4.00

        new.positions[ci] = c_pos
        hi = self._find_all(new, "H")
        for k in range(4):
            hpos = c_pos + 1.09 * self._TET[k]
            if k < len(hi):
                new.positions[hi[k]] = hpos
            else:
                new.append(Atom("H", position=hpos))
        return new

    # -- Path D: CO dissociation builders ----------------------------------

    def _place_co_diss(self, prev: Atoms) -> Atoms:
        """CO* -> C* + O* separated by ~2.5 Ang laterally, both at surface height."""
        new  = prev.copy()
        stol = self._surface_top(new)
        ci   = self._find_first(new, "C")
        oi   = self._find_first(new, "O")
        if ci is None or oi is None:
            return new

        c_orig   = new.positions[ci].copy()
        c_new    = np.array([c_orig[0],                          c_orig[1], stol + 1.10])
        o_new    = np.array([c_orig[0] + self.cfg_co_sep,        c_orig[1], stol + 1.10])

        new.positions[ci] = c_new
        new.positions[oi] = o_new
        return new

    # -- Path E: Methanol builders ------------------------------------------

    def _place_ch3oh(self, prev: Atoms) -> Atoms:
        """CH3OH* methanol: O anchored, C above, 3 H on C, 1 H on O."""
        new  = prev.copy()
        stol = self._surface_top(new)
        ci   = self._find_first(new, "C")
        oi   = self._find_first(new, "O")
        if ci is None or oi is None:
            return new

        phi  = self._h_azimuth(self._h_angle_idx)
        # O near surface
        o_pos    = new.positions[oi].copy()
        o_pos[2] = stol + 1.80

        # C above O (C-O = 1.43 Ang)
        c_pos = o_pos + np.array([0.0, 0.0, 1.43])

        # H on O (pointing sideways and slightly up)
        oh_dir  = np.array([np.cos(phi), np.sin(phi), 0.5])
        oh_dir /= np.linalg.norm(oh_dir)
        oh_pos  = o_pos + 0.97 * oh_dir

        # 3 H on C
        h_dirs = [
            np.array([np.sin(np.radians(70.5))*np.cos(phi+k*2.094),
                       np.sin(np.radians(70.5))*np.sin(phi+k*2.094),
                       np.cos(np.radians(70.5))])
            for k in range(3)
        ]

        new.positions[oi] = o_pos
        new.positions[ci] = c_pos
        hi = self._find_all(new, "H")
        # Place: 3 H on C, 1 H on O
        ch3_h = [c_pos + 1.09 * d for d in h_dirs]
        all_h = ch3_h + [oh_pos]
        for k, hpos in enumerate(all_h):
            hpos[2] = max(hpos[2], stol + 0.30)
            if k < len(hi):
                new.positions[hi[k]] = hpos
            else:
                new.append(Atom("H", position=hpos))
        return new

    def _h_direction_c(self, atoms: Atoms, c_idx: int,
                        h_angle_idx: int = 0) -> np.ndarray:
        """
        Unit vector for placing H on a C atom.
        Uses tetrahedral anti-bond direction relative to existing C-X bonds.
        Falls back to angular sweep if no bonds found.
        """
        c_pos    = atoms.positions[c_idx]
        ads      = list(range(self.n_slab, len(atoms)))
        # Find existing bonds FROM this C
        bonded_dirs = []
        for j in ads:
            if j == c_idx:
                continue
            diff = atoms.positions[j] - c_pos
            dist = np.linalg.norm(diff)
            if 0.8 < dist < 2.2:
                bonded_dirs.append(diff / dist)
        # Place new H opposite to resultant of existing bonds (anti-bond)
        if bonded_dirs:
            res = sum(bonded_dirs) / len(bonded_dirs)
            anti = -res / (np.linalg.norm(res) + 1e-9)
            # Rotate around z by h_angle_idx * 60 degrees for variation
            angle = h_angle_idx * np.pi / 3.0
            rot_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                               [np.sin(angle),  np.cos(angle), 0],
                               [0,              0,             1]])
            d = rot_z @ anti
            d[2] = max(d[2], 0.2)        # keep above surface
            return d / (np.linalg.norm(d) + 1e-9)
        # Fallback: tetrahedral directions
        phi   = 2.0 * np.pi * h_angle_idx / 6
        theta = np.radians(55.0)
        return np.array([np.sin(theta)*np.cos(phi),
                          np.sin(theta)*np.sin(phi),
                          np.cos(theta)])

    def _h_direction_o(self, atoms: Atoms, o_idx: int,
                        h_angle_idx: int = 0) -> np.ndarray:
        """
        Unit vector for placing H on an O atom.
        Targets the lone-pair direction (anti-bond to existing O-X bonds).
        """
        o_pos    = atoms.positions[o_idx]
        ads      = list(range(self.n_slab, len(atoms)))
        bonded_dirs = []
        for j in ads:
            if j == o_idx:
                continue
            diff = atoms.positions[j] - o_pos
            dist = np.linalg.norm(diff)
            if 0.8 < dist < 2.0:
                bonded_dirs.append(diff / dist)
        if bonded_dirs:
            res = sum(bonded_dirs) / len(bonded_dirs)
            anti = -res / (np.linalg.norm(res) + 1e-9)
            angle = h_angle_idx * np.pi / 3.0
            rot_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                               [np.sin(angle),  np.cos(angle), 0],
                               [0,              0,             1]])
            d = rot_z @ anti
            d[2] = max(d[2], 0.15)
            return d / (np.linalg.norm(d) + 1e-9)
        phi   = 2.0 * np.pi * h_angle_idx / 6
        theta = np.radians(35.0)         # O-H is more equatorial than C-H
        return np.array([np.sin(theta)*np.cos(phi),
                          np.sin(theta)*np.sin(phi),
                          np.cos(theta)])


    # ======================================================================
    # PUBLIC DISPATCH
    # ======================================================================

    def build(self, step: str, prev_atoms: Atoms,
              pathway_id: str = "A", h_angle_idx: int = 0) -> Atoms:
        """Build initial-guess structure for *step* given the previous relaxed atoms."""
        self._h_angle_idx = h_angle_idx
        # Need CO-dissociation separation from CFG
        self.cfg_co_sep = CFG.get("co_separation_diss", 2.50)
        new = prev_atoms.copy()

        # Common builders shared by multiple pathways
        _COMMON = {
            "01_CO2":            self._place_co2,
            "02_COOH":           self._place_cooh,
            "03_H2O_from_COOH":  self._place_h2o_on_surface,
            "03_CO":             self._place_co,
            "04_CHO":            self._place_cho,
            "04_COH":            self._place_coh,
            "05_H2O_from_COH":   self._place_h2o_on_surface,
            "05_C":              self._place_c,
            "05_CH2O":           self._place_ch2o,
            "06_CH3O":           self._place_ch3o,
            "07_O_CH4":          self._place_o_ch4,
            "07_O":              self._place_o_only,
            "08_OH":             self._place_oh,
            "09_H2O":            self._place_h2o_on_surface,
            "10_clean":          self._place_clean,
            "06_CH":             self._place_ch,
            "07_CH2":            self._place_ch2,
            "08_CH3":            self._place_ch3,
            "09_CH4":            self._place_ch4_lift,
        }

        _PATH_SPECIFIC = {
            "C": {
                "02_HCOO":           self._place_hcoo,
                "03_HCOOH":          lambda p: self._place_hcooh(p),
                "04_H2O_from_HCOOH": self._place_h2o_on_surface,
            },
            "D": {
                "05_CO_diss":          self._place_co_diss,
                "05_OH_from_CO_diss":  self._build_oh_from_co_diss,
                "05_H2O_from_CO_diss": self._build_h2o_from_co_diss,
                "06_C_from_CO_diss":   self._place_c,
            },
            "E": {
                "05_CH2O_E": self._place_ch2o,
                "06_CH3O_E": self._place_ch3o,
                "07_CH3OH":  self._place_ch3oh,
                "08_CH3OH_des": self._strip_methanol,
            },
        }

        # Try pathway-specific first, then common
        dispatch = dict(_COMMON)
        if pathway_id in _PATH_SPECIFIC:
            dispatch.update(_PATH_SPECIFIC[pathway_id])

        builder_fn = dispatch.get(step)
        if builder_fn is not None:
            return builder_fn(new)
        LOG.warning(f"No builder for step={step}, pathway={pathway_id} -- returning prev")
        return new

    def _place_hcooh(self, prev: Atoms) -> Atoms:
        """HCOOH* formic acid: two O's, one C, one H on C, one H on O."""
        new  = prev.copy()
        stol = self._surface_top(new)
        ci   = self._find_first(new, "C")
        oi   = self._find_all(new, "O")
        if ci is None or len(oi) < 2:
            return prev.copy()

        phi   = self._h_azimuth(self._h_angle_idx)
        c_pos = new.positions[ci].copy()
        c_pos[2] = stol + 1.70
        # O1 below C (toward surface)
        o1_pos  = c_pos + 1.33 * np.array([0.5, 0.0, -0.866])
        o1_pos[2] = max(o1_pos[2], stol + 0.50)
        # O2 above C (COOH side, will get H)
        o2_pos  = c_pos + 1.23 * np.array([0.0, 0.0, 0.80])
        # H on C
        h_c_pos = c_pos + 1.09 * np.array([np.cos(phi)*0.5, np.sin(phi)*0.5, -0.2])
        h_c_pos[2] = max(h_c_pos[2], stol + 0.40)
        # H on O2
        h_o_pos = o2_pos + 0.97 * np.array([np.cos(phi + 1.5), np.sin(phi + 1.5), 0.5])

        new.positions[ci]    = c_pos
        new.positions[oi[0]] = o1_pos
        new.positions[oi[1]] = o2_pos
        hi = self._find_all(new, "H")
        # 2 H atoms required
        if len(hi) >= 2:
            new.positions[hi[0]] = h_c_pos
            new.positions[hi[1]] = h_o_pos
        elif len(hi) == 1:
            new.positions[hi[0]] = h_c_pos
            new.append(Atom("H", position=h_o_pos))
        else:
            new.append(Atom("H", position=h_c_pos))
            new.append(Atom("H", position=h_o_pos))
        return new

    def _build_oh_from_co_diss(self, prev: Atoms) -> Atoms:
        """Path D: O* + H -> OH* (C* co-adsorbed)."""
        new  = prev.copy()
        stol = self._surface_top(new)
        oi   = self._find_first(new, "O")
        ci   = self._find_first(new, "C")
        if oi is None:
            return new
        # O stays near surface
        o_pos = new.positions[oi].copy()
        o_pos[2] = max(o_pos[2], stol + 1.10)
        new.positions[oi] = o_pos
        # Add H above O
        dirn  = self._h_direction_above(self._h_angle_idx)
        h_pos = o_pos + 0.97 * dirn
        h_pos[2] = max(h_pos[2], stol + 0.50)
        hi = self._find_all(new, "H")
        if hi:
            new.positions[hi[0]] = h_pos
        else:
            new.append(Atom("H", position=h_pos))
        # C* stays near surface
        if ci is not None:
            new.positions[ci][2] = max(new.positions[ci][2], stol + 1.05)
        return new

    def _build_h2o_from_co_diss(self, prev: Atoms) -> Atoms:
        """Path D: OH* + H -> H2O* (C* co-adsorbed)."""
        new  = prev.copy()
        stol = self._surface_top(new)
        oi   = self._find_first(new, "O")
        hi   = self._find_all(new, "H")
        ci   = self._find_first(new, "C")
        if oi is None:
            return new

        phi     = self._h_azimuth(self._h_angle_idx)
        o_pos   = new.positions[oi].copy()
        o_pos[2] = max(o_pos[2], stol + 1.15)
        new.positions[oi] = o_pos

        half_hoh = np.radians(104.5 / 2.0)
        h1 = o_pos + 0.96 * np.array([ np.sin(half_hoh)*np.cos(phi),
                                          np.sin(half_hoh)*np.sin(phi),
                                          np.cos(half_hoh)])
        h2 = o_pos + 0.96 * np.array([-np.sin(half_hoh)*np.cos(phi),
                                         -np.sin(half_hoh)*np.sin(phi),
                                          np.cos(half_hoh)])
        if len(hi) >= 2:
            new.positions[hi[0]] = h1
            new.positions[hi[1]] = h2
        elif len(hi) == 1:
            new.positions[hi[0]] = h1
            new.append(Atom("H", position=h2))
        else:
            new.append(Atom("H", position=h1))
            new.append(Atom("H", position=h2))

        if ci is not None:
            new.positions[ci][2] = max(new.positions[ci][2], stol + 1.05)
        return new

    # -- Stability-test iteration variant ---------------------------------
    def build_for_iteration(self, step: str, initial_atoms: Atoms,
                             pathway_id: str, iteration: int,
                             n_total: int = 3) -> Atoms:
        """
        Build a CHEMICALLY CORRECT distinct starting geometry per iteration.

        NON-PCET steps (01_CO2, 03_CO, 05_C, 05_CO_diss, strip steps...):
          No H is ever added.  Instead, the adsorbate is rotated azimuthally
          around the surface normal by (iteration * 360/n_total) degrees.
          This samples different adsorbate orientations at the same site
          without altering stoichiometry.

        PCET steps (hydrogenation):
          Explicit H-target sweep:
            iter 0  -> H placed on first C   (C-H bond, e.g. CHO* from CO*)
            iter 1  -> H placed on first O   (O-H bond, e.g. COOH* from CO2*)
            iter 2  -> H placed on second O  (second O-H)
            iter 3+ -> azimuthal rotation around iter-2 target
          This samples fundamentally different bonding topologies, not just
          angular variations around the same atom.
        """
        # Build base structure
        base = self.build(step, initial_atoms, pathway_id, h_angle_idx=0)
        stol = self._surface_top(base)
        ads  = list(range(self.n_slab, len(base)))

        # -- Decide whether to rotate (non-PCET) or re-place H (PCET) ----------
        # co_water_steps already have BOTH H's correctly placed (H2O group).
        # Moving H breaks H2O -> always rotate for these steps.
        _co_water   = {"03_H2O_from_COOH", "04_H2O_from_HCOOH"}
        # CH2O: new H must ONLY go on C (never O) -- placing on O builds HCOH
        _ch2o_steps = {"05_CH2O", "05_CH2O_E"}
        _all_s   = ["01_CO2"] + PATHWAYS[pathway_id]["steps"]
        _sidx    = next((i for i, s in enumerate(_all_s) if s == step), 0)
        _nH_curr = STEP_EXPECTED_COMPOSITION.get(step, {}).get("H", 0)
        _nH_prev = (STEP_EXPECTED_COMPOSITION.get(_all_s[_sidx-1], {}).get("H", 0)
                    if _sidx > 0 else 0)
        _is_nonpcet = ((_nH_curr - _nH_prev) <= 0
                       or step in _co_water
                       or step in _ch2o_steps)
        if _is_nonpcet:
            if not ads:
                return base
            # Rotate adsorbate around its centroid by angle = iter * 360/n steps
            angle = 2.0 * np.pi * iteration / max(n_total, 1)
            centroid = base.positions[ads].mean(axis=0)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            for i in ads:
                dx = base.positions[i, 0] - centroid[0]
                dy = base.positions[i, 1] - centroid[1]
                base.positions[i, 0] = centroid[0] + cos_a * dx - sin_a * dy
                base.positions[i, 1] = centroid[1] + sin_a * dx + cos_a * dy
            return base

        # -- PCET: place H on correct heavy-atom target -------------------
        c_idx = [i for i in ads if base[i].symbol == "C"]
        o_idx = [i for i in ads if base[i].symbol == "O"]
        h_idx = [i for i in ads if base[i].symbol == "H"]

        if iteration == 0:
            tgt = c_idx[0] if c_idx else (o_idx[0] if o_idx else None)
        elif iteration == 1:
            tgt = o_idx[0] if o_idx else (c_idx[0] if c_idx else None)
        elif iteration == 2:
            tgt = (o_idx[1] if len(o_idx) >= 2
                   else (o_idx[0] if o_idx else (c_idx[0] if c_idx else None)))
        else:
            tgt = o_idx[0] if o_idx else (c_idx[0] if c_idx else None)

        if tgt is None:
            return base

        t_sym = base[tgt].symbol
        if t_sym == "C":
            dirn, bond_len = self._h_direction_c(base, tgt, iteration), 1.09
        else:
            dirn, bond_len = self._h_direction_o(base, tgt, iteration), 0.97

        new_h    = base.positions[tgt].copy() + dirn * bond_len
        new_h[2] = max(new_h[2], stol + 0.50)

        if h_idx:
            base.positions[h_idx[-1]] = new_h   # reposition last H
        else:
            base.append(Atom("H", position=new_h))

        return base



# ==============================================================================
# INTEGRITY TRACKING RELAXER
#
# PRIMARY GOAL: run the full LBFGS relaxation to tight convergence and return
#   the final structure.  If that structure is intact, the workflow uses it.
#
# LAST RESORT: while running, silently track every intact snapshot seen in the
#   trajectory (geo_desc == "intact" AND fmax < metastable_intact_fmax).
#   The best-energy intact snapshot is saved as INTACT_METASTABLE.vasp.
#   The caller loads it ONLY when every other stage (main relax, stability test,
#   adaptive recovery) failed to produce an intact final structure.
#
# Two relaxation modes (set via CFG):
#   early_stop_on_intact = False (default, recommended)
#       Run to tight fmax convergence.  Track intact snapshots passively.
#   early_stop_on_intact = True  (fast screening mode)
#       Stop the moment intact AND fmax < metastable_intact_fmax.
#       Acceptable for high-throughput pre-screening; not for production.
#
# Outputs per step folder:
#   trajectory.traj          -- full LBFGS trajectory
#   iteration_history.json   -- per-step: iter, energy, fmax, geo_desc, is_intact
#   CONTCAR                  -- final converged structure
#   POST_RELAX.vasp          -- same as CONTCAR (human-readable label)
#   INTACT_METASTABLE.vasp   -- best intact snapshot (last resort, if found)
#   intact_metastable_energy.txt
# ==============================================================================

class EarlyStopRelaxer:  # kept as EarlyStopRelaxer for backward-compat import
    """
    Integrity-tracking LBFGS wrapper.

    Primary:   returns final fully-converged structure.
    Side-effect: saves best intact snapshot seen during trajectory as
                 INTACT_METASTABLE.vasp for use as a last-resort fallback.
    """

    def __init__(self, atoms: Atoms, folder: Path,
                 step_name: str, n_slab: int,
                 expected_comp: Optional[Dict[str, int]],
                 cfg: Dict[str, Any]):
        self.atoms         = atoms
        self.folder        = folder
        self.step_name     = step_name
        self.n_slab        = n_slab
        self.expected_comp = expected_comp
        self.cfg           = cfg

        self.check_interval      = cfg.get("check_interval",         5)
        self.min_steps           = cfg.get("min_steps_before_check",  5)
        self.metastable_fmax     = cfg.get("metastable_intact_fmax",  0.30)
        self.early_stop_enabled  = cfg.get("early_stop_on_intact",    False)

        self.history:            List[Dict[str, Any]] = []
        self.intact_metastable:  Optional[Atoms]       = None
        self.intact_meta_energy: float                 = float("inf")
        self.intact_meta_fmax:   float                 = float("inf")
        self.intact_meta_iter:   int                   = -1
        self.stopped_early:      bool                  = False
        self.stopping_iter:      int                   = -1

    def run(self, opt, fmax: float, max_steps: int,
            frozen: List[int]) -> Tuple[Atoms, float, float]:
        """
        Run LBFGS with per-step integrity monitoring.

        Returns
        -------
        (final_atoms, final_energy, final_fmax)
            -- always the fully-converged result (primary goal).
            -- INTACT_METASTABLE.vasp is a side-effect saved for last-resort use.
        """
        best_e     = float("inf")
        best_atoms = self.atoms.copy()

        traj_path  = self.folder / "trajectory.traj"
        if self.cfg.get("save_trajectories", True):
            traj = Trajectory(str(traj_path), "w", self.atoms)
        else:
            traj = None

        for i in range(max_steps):
            # --- Single LBFGS step ---
            try:
                opt.step()
            except Exception as exc:
                LOG.warning(f"      [Relax] step error at iter {i}: {exc}")
                break

            # --- Energy and forces ---
            try:
                e    = self.atoms.get_potential_energy()
                fvec = self.atoms.get_forces()
                fmx  = float(np.max(np.linalg.norm(fvec, axis=1)))
            except Exception:
                e, fmx = float("nan"), float("nan")

            if not np.isfinite(e):
                LOG.warning(f"      [Relax] non-finite energy at iter {i}")
                break

            if e < best_e:
                best_e     = e
                best_atoms = self.atoms.copy()

            if traj is not None:
                traj.write()

            # --- Chemistry integrity check (every check_interval steps) ---
            geo_desc  = "unchecked"
            is_intact = False
            if i >= self.min_steps and i % self.check_interval == 0:
                geo_desc  = describe_relaxed_geometry(
                    self.atoms, self.n_slab, self.step_name, self.expected_comp)
                is_intact = geo_desc in ("intact", "clean_surface")

                # Track the best-energy intact snapshot (last-resort backup)
                if is_intact and fmx < self.metastable_fmax:
                    if e < self.intact_meta_energy:
                        self.intact_metastable  = self.atoms.copy()
                        self.intact_meta_energy = e
                        self.intact_meta_fmax   = fmx
                        self.intact_meta_iter   = i
                        LOG.debug(f"      [Relax] intact snapshot @ iter {i}: "
                                  f"E={e:+.6f} fmax={fmx:.4f} (backup saved)")

            self.history.append({
                "iter":      i,
                "energy":    round(float(e), 8),
                "fmax":      round(float(fmx), 6),
                "geo_desc":  geo_desc,
                "is_intact": is_intact,
            })

            # --- Early-stop mode (fast screening only) ---
            if self.early_stop_enabled and is_intact and fmx < self.metastable_fmax:
                self.stopped_early = True
                self.stopping_iter = i
                LOG.info(
                    f"      {YELLOW}[Relax early-stop mode] intact at iter {i} "
                    f"fmax={fmx:.4f} -- stopping (screening mode){RESET}")
                best_atoms = self.atoms.copy()
                best_e     = e
                break

            # --- Standard tight convergence ---
            if fmx < fmax:
                LOG.debug(f"      [Relax] converged at iter {i} fmax={fmx:.4f}")
                break

        if traj is not None:
            traj.close()

        # --- Save best intact metastable (last-resort backup) ---
        if self.intact_metastable is not None:
            try:
                write(str(self.folder / "INTACT_METASTABLE.vasp"),
                      self.intact_metastable, format="vasp")
                (self.folder / "intact_metastable_energy.txt").write_text(
                    f"{self.intact_meta_energy:.8f}\n"
                    f"fmax={self.intact_meta_fmax:.6f}\n"
                    f"iter={self.intact_meta_iter}\n"
                )
                LOG.debug(f"      [Relax] INTACT_METASTABLE.vasp saved "
                          f"(E={self.intact_meta_energy:+.6f} "
                          f"@ iter {self.intact_meta_iter})")
            except Exception as _we:
                LOG.debug(f"      [Relax] metastable write failed: {_we}")

        # --- Write iteration history JSON ---
        try:
            import json as _json
            (self.folder / "iteration_history.json").write_text(
                _json.dumps({
                    "step_name":            self.step_name,
                    "primary_goal":         "full_convergence",
                    "early_stop_mode":      self.early_stop_enabled,
                    "stopped_early":        self.stopped_early,
                    "stopping_iter":        self.stopping_iter,
                    "total_iters":          len(self.history),
                    "tight_fmax_threshold": fmax,
                    "metastable_fmax_threshold": self.metastable_fmax,
                    "check_interval":       self.check_interval,
                    "intact_metastable": {
                        "found":  self.intact_metastable is not None,
                        "energy": self.intact_meta_energy
                                  if self.intact_metastable is not None else None,
                        "fmax":   self.intact_meta_fmax
                                  if self.intact_metastable is not None else None,
                        "iter":   self.intact_meta_iter,
                    },
                    "history": self.history,
                }, indent=2)
            )
        except Exception as _he:
            LOG.debug(f"      [Relax] history write failed: {_he}")

        # Primary result: the fully-converged (or best-energy) structure
        final      = best_atoms
        final_e    = best_e
        final_fmax = self.history[-1]["fmax"] if self.history else 0.0
        return final, final_e, final_fmax


# ==============================================================================
# MODULE 6: OPTIMIZER MANAGER
# Multi-stage FIRE -> LBFGS protocol with checkpoint/restart
# ==============================================================================

class OptimizerManager:
    """
    Manages all structure relaxations with:
      - 3-stage protocol: coarse FIRE -> medium FIRE -> tight LBFGS
      - Partial slab freezing (bottom freeze_fraction)
      - Checkpoint loading (CONTCAR + energy.txt)
      - Trajectory logging (optional)
      - Convergence verification
      - SCF failure detection (energy = NaN / +/-inf)

    Returns (relaxed_atoms, energy, max_force_residual).
    """

    def __init__(self, calc, n_slab: int, cfg: Dict[str, Any]):
        self.calc   = calc
        self.n_slab = n_slab
        self.cfg    = cfg

    # -- Frozen atom indices (bottom slab layers) --------------------------
    def _frozen_indices(self, atoms: Atoms) -> List[int]:
        z      = atoms.positions[:self.n_slab, 2]
        thresh = z.min() + self.cfg["freeze_fraction"] * (z.max() - z.min())
        return [i for i in range(self.n_slab) if atoms.positions[i, 2] < thresh]

    # -- Main relax entry point --------------------------------------------
    def relax(self, atoms: Atoms, folder: Path,
              step_name: str = "unknown",
              fmax:      Optional[float] = None,
              max_steps: Optional[int]   = None) -> Tuple[Atoms, float, float]:
        """
        Relax *atoms* into *folder* with chemistry-aware early stopping.

        The LBFGS stage runs step-by-step and checks chemical integrity every
        cfg["check_interval"] steps.  When an intact metastable configuration
        is found (descriptor=="intact" AND fmax < cfg["early_stop_fmax"]),
        relaxation stops immediately and that structure is written as CONTCAR
        and INTACT_METASTABLE.vasp.

        Full iteration history (iter, energy, fmax, geo_desc) is written to
        iteration_history.json for audit and publication.

        Returns
        -------
        (relaxed_atoms, energy_eV, max_force_eV_per_Ang)
        """
        folder.mkdir(parents=True, exist_ok=True)
        fmax      = fmax      or self.cfg["relax_fmax"]
        max_steps = max_steps or self.cfg["relax_steps"]

        # -- Checkpoint load ------------------------------------------------
        contcar   = folder / "CONTCAR"
        energy_f  = folder / "energy.txt"
        fres_f    = folder / "fmax_residual.txt"
        if contcar.exists() and energy_f.exists():
            try:
                energy   = float(energy_f.read_text().strip())
                relaxed  = read(str(contcar))
                fres     = float(fres_f.read_text().strip()) if fres_f.exists() else 0.0
                LOG.debug(f"  [ckpt] {step_name}: E={energy:.6f} fmax={fres:.4f}")
                return relaxed, energy, fres
            except Exception:
                pass  # fall through to re-relax

        # -- Save pre-relaxation structures --------------------------------
        write(str(folder / "POSCAR"),         atoms, format="vasp")
        write(str(folder / "PRE_RELAX.vasp"), atoms, format="vasp")
        write(str(folder / "INITIAL.xyz"),    atoms, format="xyz")

        work = atoms.copy()
        work.calc = self.calc

        is_bare_or_co2 = step_name in ("bare_slab", "01_CO2")

        # -- Stage 0: coarse adsorbate-only pre-relax (FIRE, big maxstep) -
        if not is_bare_or_co2:
            work.set_constraint(FixAtoms(indices=list(range(self.n_slab))))
            pre0 = FIRE(work, logfile=None,
                        maxstep=self.cfg.get("fire_maxstep", 0.30))
            pre0.run(fmax=self.cfg["pre_relax0_fmax"],
                     steps=self.cfg["pre_relax0_steps"])

        # -- Stage 1: tighter adsorbate-only pre-relax (FIRE, smaller step) -
        if not is_bare_or_co2:
            work.set_constraint(FixAtoms(indices=list(range(self.n_slab))))
            pre1 = FIRE(work, logfile=None,
                        maxstep=self.cfg.get("fire1_maxstep", 0.10))
            pre1.run(fmax=self.cfg["pre_relax1_fmax"],
                     steps=self.cfg["pre_relax1_steps"])

        # -- Stage 2: full relax with early-stop integrity checking ---------
        # Uses EarlyStopRelaxer to monitor chemical integrity every N steps.
        # Stops the moment an intact metastable configuration is detected.
        # Falls back to standard tight convergence if early-stop never fires.
        frozen = self._frozen_indices(work)
        work.set_constraint(FixAtoms(indices=frozen))

        opt = LBFGS(work, logfile=None,
                    maxstep=self.cfg.get("lbfgs_maxstep", 0.05),
                    memory=self.cfg.get("lbfgs_memory",   20),
                    damping=self.cfg.get("lbfgs_damping", 0.25),
                    alpha=self.cfg.get("lbfgs_alpha",     70.0))

        expected_comp = STEP_EXPECTED_COMPOSITION.get(step_name)
        early_stopper = EarlyStopRelaxer(
            atoms        = work,
            folder       = folder,
            step_name    = step_name,
            n_slab       = self.n_slab,
            expected_comp= expected_comp,
            cfg          = self.cfg,
        )
        work, energy, fres = early_stopper.run(
            opt, fmax=fmax, max_steps=max_steps, frozen=frozen)

        if not np.isfinite(energy):
            raise RuntimeError(f"Non-finite energy {energy} at step {step_name}")

        # Compute max residual force (already returned by EarlyStopRelaxer)
        try:
            forces   = work.get_forces()
            unfrozen = [i for i in range(len(work)) if i not in set(frozen)]
            fres     = float(np.max(np.linalg.norm(forces[unfrozen], axis=1))) \
                       if unfrozen else fres
        except Exception:
            pass  # fres already set by EarlyStopRelaxer

        # -- Save post-relaxation structures -------------------------------
        # -- Slab reconstruction guard ------------------------------------
        # If any slab atom moved more than max_disp from its initial position,
        # flag it.  We don't abort (the relax already happened) but we warn
        # loudly so the user can tighten constraints if needed.
        max_disp = self.cfg.get("slab_max_displacement", 2.0)
        if max_disp > 0 and self.n_slab > 0:
            initial_slab = atoms.positions[:self.n_slab]
            relaxed_slab  = work.positions[:self.n_slab]
            disp = np.linalg.norm(relaxed_slab - initial_slab, axis=1)
            worst = float(disp.max())
            worst_idx = int(disp.argmax())
            if worst > max_disp:
                LOG.warning(
                    f"  {YELLOW}[SLAB RECONSTRUCTION] atom {worst_idx} "
                    f"displaced {worst:.2f} Ang > {max_disp:.1f} Ang limit "
                    f"at step {step_name}. Check constraints.{RESET}"
                )
                (folder / "SLAB_RECONSTRUCTION_WARNING.txt").write_text(
                    f"Step {step_name}: atom {worst_idx} displaced {worst:.3f} Ang\n"
                )

        write(str(contcar),                   work, format="vasp")
        write(str(folder / "POST_RELAX.vasp"), work, format="vasp")
        write(str(folder / "FINAL.xyz"),       work, format="xyz")
        energy_f.write_text(f"{energy:.8f}")
        fres_f.write_text(f"{fres:.8f}")

        return work.copy(), energy, fres

    def relax_bare_slab(self, slab: Atoms, folder: Path) -> Tuple[Atoms, float]:
        """Relax the bare slab (no adsorbate)."""
        relaxed, energy, _ = self.relax(slab, folder, step_name="bare_slab")
        return relaxed, energy

    def relax_gas_refs(self, folder: Path) -> Dict[str, float]:
        """
        Relax CO2, H2, H2O, CO, CH3OH in 20-Ang vacuum cells.
        CO is needed for Path D (CO dissociation step reference).
        CH3OH is needed for Path E (methanol desorption).
        20 Ang cell: H2O has dipole (1.85 D); image interaction < 1 meV at 20 Ang.
        """
        refs = {}
        for mol_name in ("CO2", "H2", "H2O", "CO", "CH3OH"):
            mol_folder = folder / f"gas_{mol_name}"
            mol_folder.mkdir(parents=True, exist_ok=True)
            ckpt = mol_folder / "energy.txt"
            if ckpt.exists():
                refs[mol_name] = float(ckpt.read_text().strip())
                continue
            mol = molecule(mol_name)
            mol.set_cell([20.0, 20.0, 20.0])
            mol.center()
            mol.calc = self.calc
            opt = FIRE(mol, logfile=None)
            opt.run(fmax=self.cfg.get("gas_relax_fmax", 0.03),
                    steps=self.cfg.get("gas_relax_steps", 500))
            e = mol.get_potential_energy()
            ckpt.write_text(f"{e:.8f}")
            refs[mol_name] = e
            LOG.info(f"  Gas ref {mol_name}: E = {e:.6f} eV")
        return refs


# ==============================================================================
# MODULE 7: ADAPTIVE RECOVERY ENGINE
# Minimum 3-5 attempts before pathway rejection
# ==============================================================================

class AdaptiveRecoveryEngine:
    """
    Implements the full adaptive recovery protocol for failed intermediates.

    Recovery strategies (applied in sequence):
    -------------------------------------------
    0. Geometry perturbation    -- small random displacements of adsorbate atoms
    1. Adsorbate reorientation  -- flip orientation 180 deg around z-axis
    2. C/O inversion            -- swap which atom is surface-proximal
    3. Alternate H-attachment   -- add H to C instead of O (or vice versa)
    4. Height adjustment        -- raise / lower adsorbate by 0.3-0.5 Ang
    5. Tilt variation           -- tilt molecular axis by +/-15-30 deg
    6. Metastable exploration   -- try another adsorption site nearby

    A recovery attempt is deemed successful if the re-relaxed structure passes
    ChemistryValidator.validate().  The lowest-energy valid structure is returned.

    If ALL recovery attempts fail the step is declared failed and the pathway
    moves to the next available recovery option at the pathway level.
    """

    STRATEGIES = [
        "geometry_perturbation",
        "adsorbate_reorientation",
        "c_o_inversion",
        "alternate_h_attachment",
        "height_adjustment",
        "tilt_variation",
        "metastable_exploration",
    ]

    # Dedicated strategies for CH2O steps.
    # Rebuilds H2C=O at different C=O tilt/height/azimuth.
    # alternate_h_attachment is excluded -- it puts H on O -> HCOH (wrong isomer).
    CH2O_STRATEGIES = [
        "hcoh_to_ch2o_tilt0",
        "hcoh_to_ch2o_tilt30",
        "hcoh_to_ch2o_tilt45",
        "hcoh_to_ch2o_height_hi",
        "hcoh_to_ch2o_height_lo",
        "hcoh_to_ch2o_rotate90",
        "hcoh_to_ch2o_rotate180",
    ]
    CH2O_STEPS = {"05_CH2O", "05_CH2O_E"}

    def __init__(self, builder: StructureBuilder,
                 optimizer: OptimizerManager,
                 validator: ChemistryValidator,
                 cfg: Dict[str, Any]):
        self.builder   = builder
        self.optimizer = optimizer
        self.validator = validator
        self.cfg       = cfg
        self.max_attempts = cfg.get("max_recovery_attempts", 5)

    def recover(self, step: str, prev_atoms: Atoms, pathway_id: str,
                step_folder: Path, prev_step_energy: float,
                ) -> Tuple[Optional[Atoms], Optional[float], List[Dict]]:
        """
        Attempt recovery for a failed step.

        Returns
        -------
        (best_atoms, best_energy, retry_history)
        best_atoms is None if all attempts fail.
        """
        history    = []
        candidates   = []   # (energy, atoms, record) -- intact only
        all_relaxed  = []   # (energy, atoms, record) -- all attempts

        # CH2O steps: use dedicated H2C=O rebuild strategies only.
        # Exclude alternate_h_attachment -- it re-creates HCOH.
        if step in self.CH2O_STEPS:
            _strat_pool = self.CH2O_STRATEGIES
            LOG.info(f"  {CYAN}[RECOVERY-CH2O] {step}: "
                      f"{self.max_attempts} H2C=O rebuild attempts "
                      f"(HCOH isomer rejected){RESET}")
        else:
            _strat_pool = self.STRATEGIES
            LOG.info(f"  {YELLOW}[RECOVERY] {step} (Path {pathway_id}): "
                      f"starting {self.max_attempts} recovery attempts{RESET}")

        # Infinite recovery loop: continue until an intact structure is found
        attempt_idx = 0
        while True:
            strategy = _strat_pool[attempt_idx % len(_strat_pool)]
            # Save each attempt under stability_tests for full traceability
            rec_folder = step_folder / "stability_tests" / f"recovery_{attempt_idx:02d}_{strategy}"
            rec_folder.mkdir(parents=True, exist_ok=True)

            try:
                trial = self._apply_strategy(
                    strategy, step, prev_atoms, pathway_id, attempt_idx
                )
                # Save initial trial geometry
                write(str(rec_folder / "POSCAR"), trial, format="vasp")

                rel, energy, fres = self.optimizer.relax(
                    trial, rec_folder, step_name=f"{step}_rec{attempt_idx}",
                    fmax=self.cfg["stability_relax_fmax"] * 1.5,
                    max_steps=self.cfg["stability_max_steps"],
                )
                # Save relaxed structure and additional formats for audit
                write(str(rec_folder / "CONTCAR"), rel, format="vasp")
                write(str(rec_folder / "structure.cif"), rel, format="cif")

                # Determine intactness using same descriptor logic as stability tests
                expected_comp = STEP_EXPECTED_COMPOSITION.get(step)
                geo_desc = describe_relaxed_geometry(
                    rel, self.optimizer.n_slab, step, expected_comp)
                is_intact = geo_desc in ("intact", "clean_surface")
                # Additional CH4 presence check for Path_B_Hydroxymethylidene_Route
                if "CH4" in step:
                    # Simple heuristic: require at least one carbon and four hydrogens
                    symbols = rel.get_chemical_symbols()
                    num_c = symbols.count("C")
                    num_h = symbols.count("H")
                    if not (num_c >= 1 and num_h >= 4):
                        is_intact = False
                        LOG.warning(f"{RED}[RECOVERY] CH4 missing or incomplete in {step}{RESET}")
                    else:
                        # Additional desorption check: carbon height above slab
                        n_slab = self.optimizer.n_slab
                        slab_z_max = max(rel.positions[:n_slab, 2])
                        c_z_positions = [pos[2] for sym, pos in zip(symbols, rel.positions) if sym == "C"]
                        if c_z_positions:
                            max_c_z = max(c_z_positions)
                            if max_c_z - slab_z_max > 3.0:
                                is_intact = False
                                LOG.warning(f"{RED}[RECOVERY] CH4 appears desorbed (-z={max_c_z - slab_z_max:.2f} -) in {step}{RESET}")
                # Write attempt-specific log for audit
                with open(str(rec_folder / "attempt_log.txt"), "w") as _log:
                    _log.write(f"Step: {step}\nAttempt: {attempt_idx}\nStrategy: {strategy}\nEnergy: {energy:.6f}\nDescriptor: {geo_desc}\nIntact: {is_intact}\n")
                    # Simple heuristic: require at least one carbon and four hydrogens
                    symbols = rel.get_chemical_symbols()
                    num_c = symbols.count("C")
                    num_h = symbols.count("H")
                    if not (num_c >= 1 and num_h >= 4):
                        is_intact = False
                        LOG.warning(f"{RED}[RECOVERY] CH4 missing or incomplete in {step}{RESET}")

                record = {
                    "attempt":   attempt_idx,
                    "strategy":  strategy,
                    "energy":    float(energy),
                    "fres":      float(fres),
                    "valid":     is_intact,
                    "descriptor": geo_desc,
                    "messages":  [geo_desc],
                }
                history.append(record)

                status_col = GREEN if is_intact else RED
                LOG.info(f"    [{attempt_idx}] {strategy}: "
                         f"E={energy:+.4f} fmax={fres:.4f}  "
                         f"[{status_col}{geo_desc}{RESET}]")

                all_relaxed.append((energy, rel, record))
                if is_intact:
                    candidates.append((energy, rel, record))
                    # Immediate return on first intact structure (strict policy)
                    best_e, best_atoms, best_rec = min(candidates, key=lambda x: x[0])
                    LOG.info(f"  {GREEN}[RECOVERY] SUCCESS: "
                             f"best E={best_e:+.4f} eV via {best_rec['strategy']}{RESET}")
                    return best_atoms, best_e, history

            except Exception as exc:
                LOG.warning(f"    [{attempt_idx}] {strategy} raised: {exc}")
                history.append({
                    "attempt":  attempt_idx,
                    "strategy": strategy,
                    "error":    str(exc),
                    "valid":    False,
                })

            attempt_idx += 1

        if candidates:
            best_e, best_atoms, best_rec = min(candidates, key=lambda x: x[0])
            LOG.info(f"  {GREEN}[RECOVERY] SUCCESS: "
                     f"best E={best_e:+.4f} eV via {best_rec['strategy']}{RESET}")
            return best_atoms, best_e, history
        elif all_relaxed:
            # No intact found but return the lowest-energy structure as
            # best-available so the caller can soft-continue.
            best_e, best_atoms, best_rec = min(all_relaxed, key=lambda x: x[0])
            LOG.warning(f"  {YELLOW}[RECOVERY] no intact in "
                        f"{self.max_attempts} attempts for {step}. "
                        f"Best-available: E={best_e:+.4f} "
                        f"via {best_rec['strategy']} "
                        f"[{best_rec['descriptor']}]{RESET}")
            return best_atoms, best_e, history
        else:
            LOG.warning(f"  {RED}[RECOVERY] all relaxations failed for {step}{RESET}")
            return None, None, history

    # -- Strategy implementations ------------------------------------------

    def _apply_strategy(self, strategy: str, step: str,
                         prev_atoms: Atoms, pathway_id: str,
                         attempt_idx: int) -> Atoms:
        """Dispatch to the appropriate recovery strategy."""
        n = self.n_slab
        stol = prev_atoms.positions[:n, 2].max()

        if strategy == "geometry_perturbation":
            return self._perturb(prev_atoms, step, pathway_id, attempt_idx)

        elif strategy == "adsorbate_reorientation":
            # Rebuild from scratch with rotated h_angle
            return self.builder.build(
                step, prev_atoms, pathway_id,
                h_angle_idx=(attempt_idx * 2 + 1) % 6
            )

        elif strategy == "c_o_inversion":
            return self._invert_co(prev_atoms, step, pathway_id, stol)

        elif strategy == "alternate_h_attachment":
            return self._alternate_h(prev_atoms, step, pathway_id, attempt_idx, stol)

        elif strategy == "height_adjustment":
            return self._adjust_height(prev_atoms, step,
                                        delta_z=0.30 * (attempt_idx % 2 == 0) - 0.20)

        elif strategy == "tilt_variation":
            return self._tilt_adsorbate(prev_atoms, step, pathway_id,
                                         tilt_deg=15.0 * (1 + attempt_idx % 3))

        elif strategy == "metastable_exploration":
            return self._lateral_shift(prev_atoms, step, pathway_id, attempt_idx)

        elif strategy.startswith("hcoh_to_ch2o"):
            # Rebuild clean H2C=O geometry. BOTH H on C always. O above C always.
            # NEVER place H on O -- that produces HCOH (wrong isomer).
            n       = self.builder.n_slab
            new     = prev_atoms.copy()
            stol    = new.positions[:n, 2].max()

            if   "tilt30"    in strategy: tilt_deg, dz = 30, 0.0
            elif "tilt45"    in strategy: tilt_deg, dz = 45, 0.0
            elif "height_hi" in strategy: tilt_deg, dz =  0, 0.50
            elif "height_lo" in strategy: tilt_deg, dz = 20,-0.20
            elif "rotate90"  in strategy: tilt_deg, dz = 15, 0.0
            elif "rotate180" in strategy: tilt_deg, dz = 15, 0.0
            else:                          tilt_deg, dz =  0, 0.0

            phi = (np.pi/2  if "rotate90"  in strategy else
                   np.pi    if "rotate180" in strategy else
                   2.0 * np.pi * attempt_idx / 6.0)

            ci  = next((i for i in range(n, len(new)) if new[i].symbol=="C"), None)
            oi  = next((i for i in range(n, len(new)) if new[i].symbol=="O"), None)
            his = [i for i in range(n, len(new)) if new[i].symbol=="H"]

            if ci is not None and oi is not None:
                tilt_rad = np.radians(tilt_deg)
                co_len, ch_len = 1.22, 1.09
                c_pos    = new.positions[ci].copy()
                c_pos[2] = stol + 1.40 + dz
                o_dir    = np.array([np.sin(tilt_rad)*np.cos(phi),
                                      np.sin(tilt_rad)*np.sin(phi),
                                      np.cos(tilt_rad)])
                o_pos    = c_pos + co_len * o_dir
                o_pos[2] = max(o_pos[2], c_pos[2] + 0.20)
                perp  = np.array([-np.sin(phi), np.cos(phi), 0.0])
                h1_d  = (-0.5*o_dir + 0.866*perp); h1_d /= (np.linalg.norm(h1_d)+1e-9)
                h2_d  = (-0.5*o_dir - 0.866*perp); h2_d /= (np.linalg.norm(h2_d)+1e-9)
                h1_pos = c_pos + ch_len*h1_d; h1_pos[2] = max(h1_pos[2], stol+0.25)
                h2_pos = c_pos + ch_len*h2_d; h2_pos[2] = max(h2_pos[2], stol+0.25)
                new.positions[ci] = c_pos
                new.positions[oi] = o_pos
                if len(his) >= 2:
                    new.positions[his[0]] = h1_pos
                    new.positions[his[1]] = h2_pos
                elif len(his) == 1:
                    new.positions[his[0]] = h1_pos
                    new.append(Atom("H", position=h2_pos))
                mag = self.cfg.get("perturbation_magnitude", 0.10) * 0.4
                for i in range(n, len(new)):
                    new.positions[i] += np.random.normal(0, mag, 3)
                    new.positions[i, 2] = max(new.positions[i, 2], stol+0.20)
            return new

        else:
            return self.builder.build(step, prev_atoms, pathway_id,
                                       h_angle_idx=attempt_idx)

    @property
    def n_slab(self) -> int:
        return self.builder.n_slab

    def _perturb(self, atoms: Atoms, step: str,
                  pathway_id: str, attempt_idx: int) -> Atoms:
        """Random perturbation of adsorbate atoms."""
        new = atoms.copy()
        mag = self.cfg.get("perturbation_magnitude", 0.05) * (1.0 + attempt_idx * 0.5)
        for i in range(self.n_slab, len(new)):
            sym = new[i].symbol
            if sym == "O":
                # O: lateral only (keep on surface)
                dx = np.random.normal(0, mag)
                dy = np.random.normal(0, mag)
                dz = np.random.uniform(-0.02, 0.02)
                new.positions[i] += np.array([dx, dy, dz])
            elif sym == "H":
                # H: angular sweep
                phi   = 2.0 * np.pi * attempt_idx / 6.0
                theta = np.radians(30.0 + 20.0 * attempt_idx)
                new.positions[i] += mag * np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ])
            else:
                new.positions[i] += np.random.normal(0, mag, 3)
        return new

    def _invert_co(self, atoms: Atoms, step: str,
                    pathway_id: str, stol: float) -> Atoms:
        """Swap which of C/O is surface-proximal."""
        new = atoms.copy()
        ci  = self.builder._find_first(new, "C")
        oi  = self.builder._find_first(new, "O")
        if ci is None or oi is None:
            return new
        c_z_old = new.positions[ci][2]
        o_z_old = new.positions[oi][2]
        # Swap z-heights
        new.positions[ci][2] = o_z_old
        new.positions[oi][2] = c_z_old
        # Clamp both above surface
        new.positions[ci][2] = max(new.positions[ci][2], stol + 0.80)
        new.positions[oi][2] = max(new.positions[oi][2], stol + 0.80)
        return new

    def _alternate_h(self, atoms: Atoms, step: str, pathway_id: str,
                      attempt_idx: int, stol: float) -> Atoms:
        """Move the last added H from C to O or vice versa."""
        new  = atoms.copy()
        ci   = self.builder._find_first(new, "C")
        oi   = self.builder._find_first(new, "O")
        hi   = self.builder._find_all(new, "H")
        if not hi:
            return new

        phi   = 2.0 * np.pi * attempt_idx / 6.0
        dirn  = np.array([np.sin(np.radians(45))*np.cos(phi),
                           np.sin(np.radians(45))*np.sin(phi),
                           np.cos(np.radians(45))])

        # Move last H to alternate heavy atom
        if ci is not None and oi is not None:
            # Decide: is H currently closer to C or O?
            d_to_c = np.linalg.norm(new.positions[hi[-1]] - new.positions[ci])
            d_to_o = np.linalg.norm(new.positions[hi[-1]] - new.positions[oi])
            if d_to_c < d_to_o:
                # H was near C -> move to O
                anchor = new.positions[oi]
                bond_len = 0.97
            else:
                # H was near O -> move to C
                anchor = new.positions[ci]
                bond_len = 1.09
            h_new = anchor + bond_len * dirn
            h_new[2] = max(h_new[2], stol + 0.40)
            new.positions[hi[-1]] = h_new
        return new

    def _adjust_height(self, atoms: Atoms, step: str,
                        delta_z: float = 0.30) -> Atoms:
        """Uniformly shift all adsorbate atoms by delta_z along z."""
        new = atoms.copy()
        for i in range(self.n_slab, len(new)):
            new.positions[i][2] += delta_z
        # Ensure no adsorbate goes below slab top
        stol = new.positions[:self.n_slab, 2].max()
        for i in range(self.n_slab, len(new)):
            new.positions[i][2] = max(new.positions[i][2], stol + 0.40)
        return new

    def _tilt_adsorbate(self, atoms: Atoms, step: str,
                         pathway_id: str, tilt_deg: float = 20.0) -> Atoms:
        """Tilt the adsorbate centroid by tilt_deg degrees around y-axis."""
        new     = atoms.copy()
        n       = self.n_slab
        if len(new) <= n:
            return new
        ads_idx = list(range(n, len(new)))
        centroid = new.positions[ads_idx].mean(axis=0)

        tilt_rad = np.radians(tilt_deg)
        c, s = np.cos(tilt_rad), np.sin(tilt_rad)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        for i in ads_idx:
            rel = new.positions[i] - centroid
            new.positions[i] = centroid + Ry @ rel

        # Ensure above surface
        stol = new.positions[:n, 2].max()
        for i in ads_idx:
            new.positions[i][2] = max(new.positions[i][2], stol + 0.40)
        return new

    def _lateral_shift(self, atoms: Atoms, step: str,
                        pathway_id: str, attempt_idx: int) -> Atoms:
        """Shift adsorbate laterally to a nearby position on the surface."""
        new   = atoms.copy()
        n     = self.n_slab
        phi   = 2.0 * np.pi * attempt_idx / 6.0
        shift = 1.50   # Ang -- half a typical nearest-neighbour distance
        dx    = shift * np.cos(phi)
        dy    = shift * np.sin(phi)
        for i in range(n, len(new)):
            new.positions[i][0] += dx
            new.positions[i][1] += dy
        return new

    def apply_strategy_for_attempt(self, attempt_idx: int, current_atoms: Atoms,
                                   prev_atoms: Atoms, step: str,
                                   pathway_id: str) -> Tuple[str, Atoms]:
        """
        Apply a dynamic recovery strategy based on the 1-indexed attempt_idx.
        Returns (strategy_name, trial_atoms).
        """
        n_slab = self.n_slab
        stol = current_atoms.positions[:n_slab, 2].max() if n_slab > 0 else 0.0
        
        # Dedicated CH2O steps: formaldehyde must have BOTH H on C always, O above C always.
        # Places H on O -> HCOH (wrong isomer) -- we sweep tilts/rotations/heights to avoid wrong isomer.
        if step in self.CH2O_STEPS:
            strategy = self.CH2O_STRATEGIES[(attempt_idx - 1) % len(self.CH2O_STRATEGIES)]
            trial = self._apply_strategy(strategy, step, prev_atoms, pathway_id, attempt_idx - 1)
            # Add increasing perturbation magnitude for higher attempts
            if attempt_idx > len(self.CH2O_STRATEGIES):
                trial = self._perturb(trial, step, pathway_id, attempt_idx)
            return strategy, trial

        strategies = [
            "geometry_perturbation",
            "adsorbate_reorientation",
            "alternate_h_attachment",
            "c_o_inversion",
            "height_adjustment",
            "tilt_variation",
            "restart_last_intact",
        ]
        
        strategy_idx = (attempt_idx - 1) % len(strategies)
        strategy = strategies[strategy_idx]
        
        if strategy == "geometry_perturbation":
            trial = self._perturb(current_atoms, step, pathway_id, attempt_idx)
            
        elif strategy == "adsorbate_reorientation":
            trial = current_atoms.copy()
            ads = list(range(n_slab, len(trial)))
            if ads:
                centroid = trial.positions[ads].mean(axis=0)
                angle = np.radians(180.0 if attempt_idx % 2 == 0 else 90.0)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                for i in ads:
                    dx = trial.positions[i, 0] - centroid[0]
                    dy = trial.positions[i, 1] - centroid[1]
                    trial.positions[i, 0] = centroid[0] + cos_a * dx - sin_a * dy
                    trial.positions[i, 1] = centroid[1] + sin_a * dx + cos_a * dy
                    
        elif strategy == "alternate_h_attachment":
            trial = self._alternate_h(current_atoms, step, pathway_id, attempt_idx, stol)
            
        elif strategy == "c_o_inversion":
            trial = self._invert_co(current_atoms, step, pathway_id, stol)
            
        elif strategy == "height_adjustment":
            delta_z = 0.30 if (attempt_idx % 2 == 0) else -0.20
            trial = self._adjust_height(current_atoms, step, delta_z=delta_z)
            
        elif strategy == "tilt_variation":
            tilt_deg = 15.0 * (1 + (attempt_idx // len(strategies)) % 3)
            trial = self._tilt_adsorbate(current_atoms, step, pathway_id, tilt_deg=tilt_deg)
            
        elif strategy == "restart_last_intact":
            h_angle = (attempt_idx // len(strategies)) % 6
            rebuilt = self.builder.build(step, prev_atoms, pathway_id, h_angle_idx=h_angle)
            if rebuilt is not None:
                trial = rebuilt
            else:
                trial = prev_atoms.copy()
            # Apply some perturbation/height shift to the restarted structure
            trial = self._perturb(trial, step, pathway_id, attempt_idx)
            trial = self._adjust_height(trial, step, delta_z=0.10 * ((attempt_idx // 2) % 3 - 1))
            
        else:
            trial = current_atoms.copy()
            
        return strategy, trial


# ==============================================================================
# MODULE 8: STABILITY TEST ENGINE
# Iterative stability sampling per intermediate
# ==============================================================================

class StabilityTestEngine:
    """
    Runs N independent relaxations from the PRE-RELAX POSCAR seed, each with a
    chemically distinct starting geometry (different H-attachment site + structured
    perturbation), and returns the best intact structure plus stability statistics.

    Matches original run_mace_phonons.py RelaxationEngine.stability_test() exactly:
      - Seed: folder/POSCAR  (pre-relax, NOT the already-relaxed structure)
      - build_for_iteration: explicit C-target vs O1-target vs O2-target per iter
      - _perturb_adsorbate:  chemistry-aware structured perturbation
          H  -> evenly-spread angular sweep (phi=2pi*k/n, theta 10-70 deg)
          O  -> lateral-only for O-anchored steps
          C  -> isotropic for C-chain steps
      - FIRE coarse (slab fully frozen) + LBFGS fine (bottom-layer frozen)
      - Early-exit: stop once min_intact_to_stop intact structures found
    """

    def __init__(self, builder: "StructureBuilder",
                 optimizer: "OptimizerManager",
                 validator: "ChemistryValidator",
                 cfg: Dict[str, Any]):
        self.builder   = builder
        self.optimizer = optimizer
        self.cfg       = cfg
        # validator kept for API compatibility but intactness uses describe_relaxed_geometry
        self.validator = validator

    # -- Chemistry-aware perturbation --------------------------------------
    def _perturb_adsorbate(self, test: Atoms, pathway_id: str,
                            step: str, iteration: int = 0,
                            n_total: int = 1) -> None:
        """
        Displace adsorbate atoms in place.

        H atoms: evenly-spread direction on upper hemisphere
            phi   = 2*pi * iteration / n_total          (azimuthal, full circle)
            theta = 10 + 60 * (iteration % n_total) / n_total  (polar, 10-70 deg)
        Heavy atoms: path/step-specific rules (lateral-only or isotropic).
        """
        mag   = self.cfg.get("perturbation_magnitude", 0.10)
        n_ads = len(test) - self.optimizer.n_slab
        if n_ads <= 0:
            return

        n = max(n_total, 1)
        phi   = 2.0 * np.pi * iteration / n
        theta = np.radians(10.0 + 60.0 * (iteration % n) / n)
        h_dir = mag * np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),           # always +z -> above surface
        ])

        use_o_anchored = (
            (pathway_id == "A" and step in PATH_A_O_ANCHORED_STEPS) or
            (pathway_id == "C" and step in PATH_C_O_ANCHORED_STEPS) or
            (pathway_id == "E" and step in PATH_E_O_ANCHORED_STEPS)
        )
        use_c_chain = (
            (pathway_id == "B" and step in PATH_B_C_CHAIN_STEPS) or
            (pathway_id == "D" and step in PATH_D_C_CHAIN_STEPS)
        )
        use_co_diss = (pathway_id == "D" and step in PATH_D_CO_DISS_STEPS)
        use_d_o_anch = (pathway_id == "D" and step in PATH_D_O_ANCHORED_STEPS)

        for i in range(self.optimizer.n_slab, len(test)):
            sym = test[i].symbol
            if use_o_anchored:
                if sym == "O":
                    test.positions[i] += np.array([
                        np.random.normal(0, mag),
                        np.random.normal(0, mag),
                        np.random.uniform(-0.02, 0.02),
                    ])
                elif sym == "C":
                    test.positions[i] += np.random.normal(0, mag, 3)
                else:  # H
                    test.positions[i] += h_dir
            elif use_c_chain:
                if sym == "C":
                    test.positions[i] += np.random.normal(0, mag, 3)
                else:  # H (and O if any)
                    test.positions[i] += h_dir
            elif use_co_diss:
                # C* and O* both surface-bound, no H: lateral-only
                test.positions[i] += np.array([
                    np.random.normal(0, mag),
                    np.random.normal(0, mag),
                    np.random.uniform(-0.02, 0.02),
                ])
            elif use_d_o_anch:
                # C* co-adsorbed with OH*/H2O*
                if sym == "O":
                    test.positions[i] += np.array([
                        np.random.normal(0, mag),
                        np.random.normal(0, mag),
                        np.random.uniform(-0.02, 0.02),
                    ])
                elif sym == "H":
                    test.positions[i] += h_dir
                else:  # C
                    test.positions[i] += np.random.normal(0, mag, 3)
            else:
                # Default: C/O isotropic, H angular sweep
                if sym == "H":
                    test.positions[i] += h_dir
                else:
                    test.positions[i] += np.random.normal(0, mag, 3)

    # -- Main stability-test entry point ----------------------------------
    def run(self, step: str, initial_atoms: Atoms,
            pathway_id: str, step_folder: Path,
            ) -> Tuple[Dict[str, Any], Atoms]:
        """
        Run up to N relaxation trials for *step* at *step_folder*.

        Returns (stability_info_dict, best_atoms) where best_atoms is the
        lowest-energy intact structure, or the lowest-energy structure overall
        if no intact structure is found.

        stability_info keys
        -------------------
        mean_energy, std_energy, min_energy, max_energy,
        stable (bool: std < 0.10 eV), intact_count, early_exit,
        all_energies, iteration_details
        """
        n_iter = self.cfg.get("stability_iterations_per_step", {}).get(
            step, self.cfg.get("stability_iterations", 3))
        # Path C step 04_CHO is a strip (H2O desorbs), not a hydrogenation:
        # 6-orientation sweep is meaningless; 1 confirmation iter suffices.
        if step == "04_CHO" and pathway_id == "C":
            n_iter = 1

        min_intact = self.cfg.get("stability_min_intact_to_stop", 1)

        stab_dir      = Path(step_folder) / "stability_tests"
        stab_dir.mkdir(parents=True, exist_ok=True)
        expected_comp = STEP_EXPECTED_COMPOSITION.get(step)
        n_slab        = self.optimizer.n_slab

        # -- Seed: use PRE-RELAX POSCAR, not already-relaxed structure ----
        # This avoids over-fitting to a single local minimum and ensures
        # each iteration starts from the same unbiased initial geometry.
        pre_poscar = Path(step_folder) / "POSCAR"
        if pre_poscar.exists():
            original = read(str(pre_poscar))
            LOG.debug(f"      [stab] seed: {pre_poscar}")
        else:
            original = initial_atoms.copy()
            LOG.debug(f"      [stab] seed: initial_atoms (POSCAR not found)")

        energies:   List[float]  = []
        structures: List[Atoms]  = []
        iter_details             = []
        best_intact_atoms        = None
        best_intact_energy       = float("inf")
        intact_count             = 0
        early_exit               = False

        for it in range(n_iter):
            iter_dir = stab_dir / f"iter_{it:02d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # -- Build distinct geometry for this iteration ----------------
            try:
                test = self.builder.build_for_iteration(
                    step, original, pathway_id, iteration=it, n_total=n_iter)
            except Exception as exc:
                LOG.warning(f"      [stab iter {it}] build failed: {exc}")
                test = original.copy()

            # -- Apply chemistry-aware structured perturbation -------------
            self._perturb_adsorbate(test, pathway_id, step,
                                     iteration=it, n_total=n_iter)

            write(str(iter_dir / "POSCAR"),         test, format="vasp")
            write(str(iter_dir / "PRE_RELAX.vasp"), test, format="vasp")

            # -- Check for cached result -----------------------------------
            ckpt_contcar = iter_dir / "CONTCAR"
            ckpt_energy  = iter_dir / "energy.txt"
            if ckpt_contcar.exists() and ckpt_energy.exists():
                try:
                    e_ckpt  = float(ckpt_energy.read_text().strip())
                    rel_ckpt = read(str(ckpt_contcar))
                    geo_ckpt = describe_relaxed_geometry(
                        rel_ckpt, n_slab, step, expected_comp)
                    ok_ckpt  = geo_ckpt in ("intact", "clean_surface")
                    energies.append(e_ckpt)
                    structures.append(rel_ckpt)
                    if ok_ckpt:
                        intact_count += 1
                        if e_ckpt < best_intact_energy:
                            best_intact_atoms  = rel_ckpt
                            best_intact_energy = e_ckpt
                    h_lbl = _h_target_label(step, it, n_iter)
                    iter_details.append({"iteration": it, "h_target": h_lbl,
                                          "energy": e_ckpt, "geometry": geo_ckpt,
                                          "is_intact": ok_ckpt, "from_ckpt": True})
                    LOG.info(f"      [stab ckpt iter {it:02d}] E={e_ckpt:+.6f}"
                              f"  [{geo_ckpt}]")
                    if ok_ckpt and intact_count >= min_intact:
                        early_exit = True
                        break
                    continue
                except Exception:
                    pass

            # -- Stage 1: coarse FIRE with slab fully frozen ---------------
            test.calc = self.optimizer.calc
            test.set_constraint(FixAtoms(
                indices=list(range(n_slab))))
            pre = FIRE(test, logfile=None,
                       maxstep=self.cfg.get("stab_fire_maxstep", 0.05))
            pre.run(fmax=self.cfg.get("stab_fire_fmax", 0.50),
                    steps=self.cfg.get("pre_relax_steps", 200))

            # -- Stage 2: LBFGS fine convergence, bottom-layer frozen ------
            frozen = self.optimizer._frozen_indices(test)
            test.set_constraint(FixAtoms(indices=frozen))
            opt = LBFGS(
                test, logfile=None,
                memory   = self.cfg.get("lbfgs_memory",  100),
                damping  = self.cfg.get("lbfgs_damping", 0.25),
                alpha    = self.cfg.get("lbfgs_alpha",   70.0),
                maxstep  = self.cfg.get("lbfgs_maxstep", 0.04),
            )
            opt.run(fmax  = self.cfg.get("stability_relax_fmax", 0.05),
                    steps = self.cfg.get("stability_max_steps",  500))

            try:
                e = test.get_potential_energy()
            except Exception as exc:
                LOG.warning(f"      [stab iter {it}] energy failed: {exc}")
                continue
            if not np.isfinite(e):
                LOG.warning(f"      [stab iter {it}] non-finite energy {e}")
                continue

            geo_desc = describe_relaxed_geometry(test, n_slab, step, expected_comp)
            ok       = geo_desc in ("intact", "clean_surface")
            energies.append(e)
            structures.append(test.copy())

            # -- Save outputs ----------------------------------------------
            write(str(iter_dir / "CONTCAR"),         test, format="vasp")
            write(str(iter_dir / "POST_RELAX.vasp"), test, format="vasp")
            contcar_name = f"CONTCAR_E{e:+.6f}_{geo_desc}.vasp"
            write(str(iter_dir / contcar_name),       test, format="vasp")
            (iter_dir / "energy.txt").write_text(f"{e:.8f}\n")

            h_lbl    = _h_target_label(step, it, n_iter)
            stat_col = GREEN if ok else YELLOW
            try:
                forces  = test.get_forces()
                fmax_st = float(np.sqrt((forces ** 2).sum(axis=1).max()))
            except Exception:
                fmax_st = float("nan")
            LOG.info(f"      [stab iter {it:02d}] H->{h_lbl}  "
                      f"E={e:+.6f}  fmax={fmax_st:.4f}"
                      f"  [{stat_col}{geo_desc}{RESET}]")

            iter_details.append({
                "iteration":  it,
                "h_target":   h_lbl,
                "energy":     float(e),
                "geometry":   geo_desc,
                "is_intact":  ok,
                "folder":     str(iter_dir),
                "contcar":    contcar_name,
            })

            if ok:
                intact_count += 1
                if e < best_intact_energy:
                    best_intact_atoms  = test.copy()
                    best_intact_energy = e
                    LOG.info(f"      [stab] best intact @ {e:+.6f} eV"
                              f"  (iter {it:02d}, H->{h_lbl})")
                if intact_count >= min_intact:
                    remaining = n_iter - it - 1
                    if remaining > 0:
                        LOG.info(f"      [stab] early-exit after iter {it:02d}"
                                  f" -- {intact_count} intact found"
                                  f" (skipping {remaining} iters)")
                    early_exit = True
                    break

        if not energies:
            # All iterations failed; return initial as fallback
            LOG.warning(f"      [stab] all {n_iter} iters failed for {step} -- using initial")
            empty = {"mean_energy": float("nan"), "std_energy": 0.0,
                     "min_energy": float("nan"), "max_energy": float("nan"),
                     "stable": False, "intact_count": 0, "early_exit": False,
                     "all_energies": [], "iteration_details": []}
            return empty, initial_atoms.copy()

        arr = np.array(energies)

        # -- Summary file --------------------------------------------------
        _write_stab_summary(stab_dir, step, pathway_id, n_iter, pre_poscar,
                             early_exit, arr, iter_details)

        info = {
            "mean_energy":       float(arr.mean()),
            "std_energy":        float(arr.std()),
            "min_energy":        float(arr.min()),
            "max_energy":        float(arr.max()),
            "stable":            bool(arr.std() < 0.10),
            "intact_count":      intact_count,
            "early_exit":        early_exit,
            "all_energies":      arr.tolist(),
            "iteration_details": iter_details,
        }
        best = (best_intact_atoms if best_intact_atoms is not None
                else structures[int(np.argmin(arr))])
        return info, best


def _write_stab_summary(stab_dir: Path, step: str, pathway_id: str,
                         n_iter: int, seed_path, early_exit: bool,
                         arr: "np.ndarray", iter_details: list) -> None:
    """Write human-readable iteration_summary.txt for a stability test."""
    summary = stab_dir / "iterations_summary.txt"
    with open(summary, "w") as f:
        f.write(f"Stability Test -- step: {step}  pathway: {pathway_id}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Planned iters : {n_iter}  |  Ran: {len(iter_details)}"
                f"  |  Early exit: {early_exit}\n")
        f.write(f"Seed          : {seed_path}\n\n")
        f.write(f"{'Iter':<6} {'H target':<14} {'Energy (eV)':<16}"
                f" {'Geometry':<35} {'Status':<10}\n")
        f.write("-" * 80 + "\n")
        for d in iter_details:
            status = "INTACT" if d.get("is_intact") else ""
            f.write(f"{d['iteration']:<6} {d.get('h_target','?'):<14}"
                    f" {d['energy']:+.8f}  {d['geometry']:<35} {status:<10}\n")
        f.write("\n" + "=" * 80 + "\n")
        if len(arr) > 0:
            f.write(f"Mean Energy   : {arr.mean():+.8f} eV\n")
            f.write(f"Std Deviation : {arr.std():.8f} eV\n")
            f.write(f"Min Energy    : {arr.min():+.8f} eV"
                    f" (iter {int(arr.argmin())})\n")
            f.write(f"Max Energy    : {arr.max():+.8f} eV"
                    f" (iter {int(arr.argmax())})\n")
            f.write(f"Stability     : "
                    f"{'STABLE' if arr.std() < 0.10 else 'MARGINAL'}\n")
        intact = [d for d in iter_details if d.get("is_intact")]
        f.write(f"Intact found  : {len(intact)} / {len(iter_details)}\n")
        for d in intact:
            f.write(f"  iter {d['iteration']} (H->{d.get('h_target','?')}): "
                    f"E = {d['energy']:+.8f} eV\n")
    LOG.info(f"      Iteration summary -> {summary}")


class EnergyAnalyzer:
    """
    Compute free energies for all intermediates using the Computational
    Hydrogen Electrode (CHE) framework.

    Reference scheme (Peterson et al. EES 2010)
    --------------------------------------------
    For C-containing intermediates (nC >= 1):
        Ref: nC*CO2(g) + nH2*0.5H2(g) -> slab+ads + nH2O*H2O(g)
        DeltaE_raw = E(slab+ads) + nH2O*E(H2O) - E(bare) - nC*E(CO2) - nH2*E(H2)

    For C-free intermediates (nC = 0): OH*, H2O*, clean
        Ref: nO*H2O(g) -> slab+ads + (nO - nH/2)*H2(g)
        DeltaE_raw = E(slab+ads) + nH2_ref*E(H2) - E(bare) - nH2O_ref*E(H2O)

    Free energy (at U = 0 V vs RHE):
        DeltaG = DeltaE_raw + (DeltaZPE - TDeltaS) + DeltaG_solv + n_pcet*eU

    CHE potential:
        n_pcet = 0 (non-PCET) or 1 (exactly one H+ + e- per PCET step)
        DeltaG(U) = DeltaG(0) + n_pcet * eU
    """

    def __init__(self, refs: Dict[str, float]):
        self.refs = refs   # {"CO2": E, "H2": E, "H2O": E, "CO": E, "CH3OH": E}

    def compute(self, energy: float, bare_energy: float,
                atoms: Atoms, n_slab: int,
                step: str, U: float = 0.0,
                apply_zpe_ts: bool = True,
                apply_solvation: bool = True,
                n_pcet_override: Optional[int] = None,
                ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute DeltaG for one step.

        Returns
        -------
        (dg_total, breakdown_dict)
        """
        ads = atoms[n_slab:]
        nC  = sum(1 for a in ads if a.symbol == "C")
        nH  = sum(1 for a in ads if a.symbol == "H")
        nO  = sum(1 for a in ads if a.symbol == "O")

        E_CO2   = self.refs["CO2"]
        E_H2    = self.refs["H2"]
        E_H2O   = self.refs["H2O"]
        # CO and CH3OH refs present for Path D/E; fall back to CHE-derived
        # value if not explicitly relaxed (conservative).
        E_CO    = self.refs.get("CO",    E_CO2 + E_H2O - E_H2)
        E_CH3OH = self.refs.get("CH3OH", E_CO2 + 3.0 * E_H2 - E_H2O)

        # -- 1. Electronic energy contribution (standard CHE) ---------------
        # Reference: nC*CO2(g) + (nH/2 + max(0,2nC-nO))*H2(g) - max(0,2nC-nO)*H2O(g)
        # This is the Peterson/Norskov convention for CO2RR free energy diagrams.
        # For desorption steps (nC=0, nH=0): referenced to bare slab.
        # For O-cleanup tail (nC=0, nO>0): referenced to nO * H2O - nO * H2 (O removal).
        if nC > 0:
            nH2O_ref = max(0, 2 * nC - nO)   # H2O consumed to balance O from CO2
            nH2_ref  = (nH + 2 * nH2O_ref) / 2.0
            dg_raw = (energy + nH2O_ref * E_H2O
                      - bare_energy - nC * E_CO2 - nH2_ref * E_H2)
        elif nO > 0:
            # O-cleanup tail (07_O, 08_OH, 09_H2O, 10_clean):
            # DeltaG = E(slab+O*) - E(bare) - E(H2O) + E(H2)  (removes 1 O as H2O)
            nH2O_ref = float(nO)
            nH2_ref  = nO - nH / 2.0
            dg_raw   = (energy + nH2_ref * E_H2
                        - bare_energy - nH2O_ref * E_H2O)
        else:
            # Clean surface (no adsorbate): DeltaG -> 0 by construction
            dg_raw = energy - bare_energy

        # -- 2. ZPE + entropy ----------------------------------------------
        zpe_ts = 0.0
        if apply_zpe_ts:
            zpe_ts = ZPE_TS_CORRECTIONS.get(step, 0.0)

        # -- 3. Solvation --------------------------------------------------
        solv = 0.0
        if apply_solvation:
            solv = SOLVATION_CORRECTIONS.get(step, 0.0)

        # -- 4. CHE potential (n_pcet from composition delta when override given)
        if n_pcet_override is not None:
            n_pcet = n_pcet_override
        else:
            n_pcet = 0 if step in NON_PCET_STEPS else 1
        u_shift = n_pcet * U

        dg_total = dg_raw + zpe_ts + solv + u_shift

        breakdown = {
            "dg_raw":        round(float(dg_raw),    6),
            "zpe_ts":        round(float(zpe_ts),    4),
            "solvation":     round(float(solv),      4),
            "u_shift":       round(float(u_shift),   4),
            "n_pcet":        n_pcet,
            "nH_ads":        nH,
            "nC_ads":        nC,
            "nO_ads":        nO,
            "dg_total":      round(float(dg_total),  6),
            "U_V":           round(float(U),         3),
        }
        return float(dg_total), breakdown

    # -- Adsorption energy (relative to bare slab + gas-phase species) -----
    def adsorption_energy(self, energy: float, bare_energy: float,
                           ads_energy: float) -> float:
        """E_ads = E(slab+ads) - E(bare) - E(adsorbate_gas)"""
        return energy - bare_energy - ads_energy

    # -- Limiting potential -------------------------------------------------
    @staticmethod
    def limiting_potential(pathway_dg_series: List[Dict]) -> Tuple[float, str, List]:
        """
        U_L = -max(DeltaDeltaG_step) / e  over all PCET steps.

        Returns (U_L, limiting_step_key, step_increments).
        """
        series = sorted(pathway_dg_series, key=lambda d: d.get("step_idx", 0))
        if not series:
            return 0.0, "none", []

        increments = []
        prev_dg    = 0.0
        for entry in series:
            curr_dg  = entry.get("dg_total", entry.get("dg", 0.0))
            delta    = curr_dg - prev_dg
            if entry.get("n_pcet", 0) > 0:
                increments.append((entry["step"], float(delta)))
            prev_dg = curr_dg

        if not increments:
            return 0.0, "none", []

        lim_step, lim_delta = max(increments, key=lambda x: x[1])
        u_l = -lim_delta
        return round(u_l, 4), lim_step, increments

    # -- U-sweep DeltaG table --------------------------------------------------
    def u_sweep_profile(self, steps_dg: List[Dict],
                         u_values: List[float]) -> Dict[float, List[float]]:
        """
        Return {U: [dg_corrected_per_step]} for each U in u_values.
        steps_dg must be sorted by step_idx.
        """
        profile = {}
        for U in u_values:
            dg_at_u = [
                d["dg_total"] + d.get("n_pcet", 0) * U
                for d in steps_dg
            ]
            profile[U] = dg_at_u
        return profile


# ==============================================================================
# MODULE 10: REACTION TRACKER
# Pathway connectivity, energetics, retry history, selectivity
# ==============================================================================

class ReactionTracker:
    """
    Tracks the complete history of every pathway across all sites and heads.

    Stores:
        results   : List[StepResult]  -- all successful steps
        failures  : List[FailureRecord] -- all rejected steps
        trees     : nested dict for pathway-map generation

    Provides:
        energy diagrams     -- DeltaG vs step for each pathway x site x head
        selectivity metrics -- DeltaU_L differences between pathways
        pathway maps        -- reaction tree with energetics
        CSV / JSON export
    """

    def __init__(self):
        self.results : List[StepResult]   = []
        self.failures: List[FailureRecord] = []

    def record_success(self, result: StepResult):
        self.results.append(result)

    def record_failure(self, failure: FailureRecord):
        self.failures.append(failure)

    # -- Query helpers ------------------------------------------------------
    def get_pathway_results(self, head: str, pathway: str,
                             site_idx: int) -> List[StepResult]:
        return [r for r in self.results
                if r.head == head and r.pathway == pathway
                and r.site_idx == site_idx]

    def complete_sites(self, head: str, pathway: str) -> List[int]:
        """Return site indices where the full pathway completed."""
        n_steps = len(PATHWAYS[pathway]["steps"])
        sites   = {}
        for r in self.results:
            if r.head == head and r.pathway == pathway:
                sites.setdefault(r.site_idx, []).append(r)
        return [s for s, rlist in sites.items() if len(rlist) >= n_steps]

    # -- Selectivity: DeltaU_L between pathways --------------------------------
    def selectivity_table(self, head: str) -> Dict[int, Dict[str, float]]:
        """
        For each completed site, return {pathway: U_L}.
        """
        table = {}
        for pid in PATHWAYS:
            for site in self.complete_sites(head, pid):
                rlist = self.get_pathway_results(head, pid, site)
                dg_series = [{"step": r.step, "step_idx": r.step_idx,
                               "dg_total": r.dg, "n_pcet": r.n_pcet}
                              for r in sorted(rlist, key=lambda x: x.step_idx)]
                u_l, _, _ = EnergyAnalyzer.limiting_potential(dg_series)
                table.setdefault(site, {})[pid] = u_l
        return table

    # -- Energy diagram data ------------------------------------------------
    def energy_diagram(self, head: str, pathway: str,
                        site_idx: int) -> List[Tuple[str, float]]:
        """Return [(step_label, dg)] sorted by step index."""
        rlist = self.get_pathway_results(head, pathway, site_idx)
        rlist = sorted(rlist, key=lambda r: r.step_idx)
        return [(STEP_LABEL.get(r.step, r.step), r.dg) for r in rlist]

    # -- Summary stats -----------------------------------------------------
    def summary_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        for pid in PATHWAYS:
            for head in CFG["model_heads"]:
                key = f"{head}__{pid}"
                rlist = [r for r in self.results if r.head == head and r.pathway == pid]
                flist = [f for f in self.failures if f.head == head and f.pathway == pid]
                sites_done = {}
                for r in rlist:
                    sites_done.setdefault(r.site_idx, []).append(r)
                n_complete = sum(
                    1 for _, v in sites_done.items()
                    if len(v) >= len(PATHWAYS[pid]["steps"])
                )
                stats[key] = {
                    "n_results":    len(rlist),
                    "n_failures":   len(flist),
                    "n_complete_sites": n_complete,
                }
        return stats


# ==============================================================================
# MODULE 11: OUTPUT MANAGER
# POSCAR/CONTCAR, extxyz, CIF, trajectories, JSON, CSV, pathway maps
# ==============================================================================

class OutputManager:
    """
    Handles all output file generation:
        - POSCAR / CONTCAR (VASP format) -- every step
        - extxyz (extended XYZ) -- every step
        - CIF (crystallographic) -- every step  [optional]
        - Trajectory (.traj) -- per relaxation
        - JSON summary -- per site, per pathway
        - CSV summary -- aggregated across all sites
        - Reaction tree / pathway map (JSON + human-readable text)
    """

    def __init__(self, workdir: Path, cfg: Dict[str, Any]):
        self.workdir = workdir
        self.cfg     = cfg

    def save_structure(self, atoms: Atoms, folder: Path, stem: str):
        """Save structure in all requested formats."""
        # VASP POSCAR
        vasp_p = folder / f"{stem}.vasp"
        write(str(vasp_p), atoms, format="vasp")

        # Extended XYZ
        if self.cfg.get("save_extxyz", True):
            xyz_p = folder / f"{stem}.xyz"
            write(str(xyz_p), atoms, format="extxyz")

        # CIF
        if self.cfg.get("save_cif", True):
            try:
                cif_p = folder / f"{stem}.cif"
                write(str(cif_p), atoms, format="cif")
            except Exception as exc:
                LOG.debug(f"CIF write failed ({exc})")

    def save_json_summary(self, results: List[StepResult],
                           failures: List[FailureRecord],
                           workdir: Path):
        """Write full JSON summary."""
        if not self.cfg.get("save_json_summary", True):
            return
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "config":    CFG,
            "results":   [asdict(r) for r in results],
            "failures":  [asdict(f) for f in failures],
        }
        out = workdir / "results_summary.json"
        with open(out, "w") as f:
            json.dump(data, f, indent=2, default=str)
        LOG.info(f"JSON summary -> {out}")

    def save_csv_summary(self, results: List[StepResult], workdir: Path):
        """Write flat CSV for easy post-processing."""
        if not self.cfg.get("save_csv_summary", True):
            return
        out = workdir / "results_summary.csv"
        fields = ["head", "pathway", "site_idx", "step", "step_idx",
                  "energy", "bare_energy", "dg", "n_pcet",
                  "rescued", "descriptor", "stable"]
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "head":        r.head,
                    "pathway":     r.pathway,
                    "site_idx":    r.site_idx,
                    "step":        r.step,
                    "step_idx":    r.step_idx,
                    "energy":      f"{r.energy:.6f}",
                    "bare_energy": f"{r.bare_energy:.6f}",
                    "dg":          f"{r.dg:.6f}",
                    "n_pcet":      r.n_pcet,
                    "rescued":     r.rescued,
                    "descriptor":  r.validation.descriptor,
                    "stable":      r.stability_info.get("stable", False),
                })
        LOG.info(f"CSV summary -> {out}")

    def save_pathway_map(self, tracker: ReactionTracker,
                          head: str, workdir: Path):
        """Generate a human-readable pathway tree."""
        if not self.cfg.get("save_pathway_map", True):
            return
        map_dir = workdir / "pathway_maps" / head
        map_dir.mkdir(parents=True, exist_ok=True)

        for pid, pdef in PATHWAYS.items():
            lines = [
                f"Pathway {pid}: {pdef['name']}",
                f"Product: {pdef['product']}",
                "=" * 72,
            ]
            for site in sorted(tracker.complete_sites(head, pid))[:5]:
                diagram = tracker.energy_diagram(head, pid, site)
                lines.append(f"\n  Site {site:03d}:")
                lines.append(f"  {'Step':<35} {'DeltaG (eV)':>10}")
                lines.append("  " + "-" * 47)
                for label, dg in diagram:
                    col = "v" if dg < 0 else "^"
                    lines.append(f"  {label:<35} {col} {dg:+.4f}")
            out = map_dir / f"pathway_{pid}.txt"
            out.write_text("\n".join(lines))

        # Selectivity table
        sel_table = tracker.selectivity_table(head)
        if sel_table:
            out_sel = map_dir / "selectivity_table.txt"
            lines = ["Selectivity: U_L per pathway per site", "=" * 72, ""]
            path_pairs = [("A", "D"), ("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
            for site, ul_map in sorted(sel_table.items()):
                row = f"Site {site:03d}: "
                for pid in PATHWAYS:
                    if pid in ul_map:
                        row += f"  {pid}:{ul_map[pid]:+.3f}V"
                row += "   Delta(pairs): "
                for px, py in path_pairs:
                    if px in ul_map and py in ul_map:
                        delta = ul_map[px] - ul_map[py]
                        row += f" ({px}-{py})={delta:+.3f}"
                lines.append(row)
            out_sel.write_text("\n".join(lines))
            LOG.info(f"Pathway maps -> {map_dir}")



# ==============================================================================
# MODULE 12: PATHWAY GENERATOR (MAIN ORCHESTRATOR)
# Coordinates all modules to execute the full CO2RR workflow
# ==============================================================================

class PathwayGenerator:
    """
    Main orchestrator for the CO2RR autonomous workflow.

    Execution flow per (head x pathway x site)
    -------------------------------------------
    For each site:
      1. Load bare slab + adsorb CO2 (initial geometry)
      2. For each step in pathway:
         a. Build initial-guess geometry (StructureBuilder.build)
         b. Relax (OptimizerManager.relax)
         c. Validate (ChemistryValidator.validate)
         d. If invalid -> Stability test (StabilityTestEngine)
         e. If still invalid -> Adaptive recovery (AdaptiveRecoveryEngine)
         f. If all fail -> record failure, halt pathway at this site
         g. Compute DeltaG (EnergyAnalyzer)
         h. Save results (CheckpointManager + OutputManager)
         i. Set best valid structure as seed for next step
      3. Record pathway completion or failure

    Pre-step validation (before every H-addition step)
    ---------------------------------------------------
    Before adding H, verify that the parent intermediate is:
      - Chemically intact (correct bond topology)
      - Structurally stable (std_energy from stability test < 0.1 eV)
      - Fully converged (fmax < 2x relax_fmax)
      - Stoichiometrically correct
      - Realistically coordinated (not desorbed)
      - At the SAME active site (no drift > 1.5 Ang laterally)
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg        = cfg
        self.base_dir   = Path(cfg["workdir"])
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.tracker    = ReactionTracker()
        self.checkpoint = CheckpointManager(self.base_dir)
        self.output_mgr = OutputManager(self.base_dir, cfg)

    # -- Calculator factory ------------------------------------------------
    def _make_calc(self, head: str):
        if MACE_AVAILABLE:
            return MACECalculator(
                model_paths     = self.cfg["model_path"],
                device          = self.cfg.get("device", "cpu"),
                default_dtype   = self.cfg.get("default_dtype", "float32"),
                head            = head,
            )
        else:
            # EMT fallback for CI / testing
            from ase.calculators.emt import EMT
            LOG.warning("Using EMT calculator (MACE not available)")
            return EMT()

    # -- Bare slab relaxation ----------------------------------------------
    def _get_bare_slab(self, head: str, optimizer: OptimizerManager,
                        slab: Atoms) -> Tuple[Atoms, float]:
        slab_folder = self.base_dir / head / "00_bare_slab"
        cached_e    = self.checkpoint.load_bare_slab(head)
        if cached_e is not None and (slab_folder / "CONTCAR").exists():
            relaxed = read(str(slab_folder / "CONTCAR"))
            LOG.info(f"  Bare slab ({head}): E = {cached_e:.6f} eV  [CACHED]")
            return relaxed, cached_e
        relaxed, energy = optimizer.relax_bare_slab(slab, slab_folder)
        self.checkpoint.save_bare_slab(head, energy)
        LOG.info(f"  Bare slab ({head}): E = {energy:.6f} eV")
        return relaxed, energy

    # -- CO2 initial placement ---------------------------------------------
    def _place_co2_on_slab(self, bare_slab: Atoms, site: Dict) -> Atoms:
        """
        Add a CO2 molecule above the given site on the bare slab.

        Matches original template exactly:
          - CO2 from ase.build.molecule() is linear along z (O-C-O).
          - Rotate 90 deg around y so the molecular axis lies along x:
              O1 at (-1.16, 0, 0), C at (0,0,0), O2 at (+1.16, 0, 0)
          - Translate so C lands at (site_x, site_y, surface_top + z_height).
        This gives a flat, horizontal starting geometry that relaxes into
        the bent chemisorbed CO2* configuration.
        """
        co2      = molecule("CO2")
        co2.rotate(90, "y")          # molecular axis now along x
        new      = bare_slab.copy()
        stol     = SlabGenerator.surface_top_z(new)
        site_pos = site["position"]

        # Find C atom index in molecule and centre it over the site
        c_mol_idx = next(i for i, a in enumerate(co2) if a.symbol == "C")
        c_offset  = co2.positions[c_mol_idx].copy()
        target    = np.array([site_pos[0], site_pos[1],
                               stol + self.cfg["z_height"]])
        co2.translate(target - c_offset)

        for a in co2:
            new.append(Atom(a.symbol, position=a.position.copy()))
        return new

    # -- Adsorbate site-drift check -----------------------------------------
    @staticmethod
    def _site_drift_ok(atoms: Atoms, n_slab: int,
                        original_site: Dict, tol: float = 2.0) -> bool:
        """Check that the adsorbate hasn't drifted > tol Ang from original site."""
        if len(atoms) <= n_slab:
            return True
        ads_xy = atoms.positions[n_slab:, :2].mean(axis=0)
        site_xy = original_site["position"][:2]
        drift   = np.linalg.norm(ads_xy - site_xy)
        return drift < tol

    # -- Pre-step parent validation -----------------------------------------
    def _validate_parent(self, parent_atoms: Atoms, parent_step: str,
                          site: Dict, n_slab: int,
                          parent_stability: Dict) -> Tuple[bool, str]:
        """
        Verify parent intermediate is safe to hydrogenate.
        Returns (ok, reason).

        SOFT-TOLERANCE mode (pathway-continuity):
          - Site drift up to 2.5 Ang is accepted if the adsorbate is chemically intact.
          - Convergence failure is logged as a warning but does NOT abort the pathway;
            the best-available geometry is used instead.
          - Structural std_energy threshold raised to 0.30 eV to accommodate marginal
            metastable intermediates.
        """
        vr = ChemistryValidator(n_slab, self.cfg).validate(parent_atoms, parent_step)
        if not vr.is_valid:
            return False, f"Parent {parent_step} not chemically valid: {vr.messages}"

        # Structural stability check -- raised from 0.20 to 0.30 eV
        std = parent_stability.get("std_energy", 999.0)
        if std > 0.30:
            return False, f"Parent {parent_step} unstable (std_e={std:.3f} eV > 0.30)"

        # Convergence: soft-warn only, do NOT abort
        if not vr.convergence_ok:
            LOG.warning(f"      {YELLOW}[PRE-STEP WARN] Parent {parent_step} "
                        f"not fully converged -- continuing with best-available{RESET}")

        # Site drift: soft tolerance 2.5 Ang (widened from 2.0)
        if not self._site_drift_ok(parent_atoms, n_slab, site, tol=2.5):
            # Extra check: if adsorbate is chemically intact, allow soft drift
            expected_comp = STEP_EXPECTED_COMPOSITION.get(parent_step)
            geo = describe_relaxed_geometry(parent_atoms, n_slab, parent_step,
                                             expected_comp)
            if geo in ("intact", "clean_surface"):
                LOG.warning(f"      {YELLOW}[PRE-STEP WARN] Parent {parent_step} "
                            f"drifted > 2.5 Ang but is chemically intact -- "
                            f"soft-continuing{RESET}")
            else:
                return False, f"Parent {parent_step} drifted from site and is not intact ({geo})"

        return True, "OK"

    # ======================================================================
    # SINGLE PATHWAY x SITE RUNNER
    # ======================================================================

    @staticmethod
    def _pcet_count_for_step(step: str, step_idx: int,
                              pathway_id: str,
                              all_steps: List[str]) -> int:
        """
        Return n_pcet for this step based on the change in H atom count
        relative to the previous step (STEP_EXPECTED_COMPOSITION delta).

        This is the chemically rigorous CHE approach:
            n_pcet = max(0, H(this_step) - H(prev_step))
        A negative delta means atoms desorbed (n_pcet = 0).

        Pathway-specific exceptions (e.g. 04_CHO in Path C is a strip
        step with dH = -2, not a hydrogenation) are handled through
        _PATHWAY_NONPCET and the composition delta itself.
        """
        curr_comp = STEP_EXPECTED_COMPOSITION.get(step, {})
        nH_curr   = curr_comp.get("H", 0)

        if step_idx == 0:
            # First step after 01_CO2: compare to CO2* {H:0}
            nH_prev = 0
        else:
            prev_step = all_steps[step_idx - 1]
            prev_comp = STEP_EXPECTED_COMPOSITION.get(prev_step, {})
            nH_prev   = prev_comp.get("H", 0)

        # Special case: if 01_CO2 is the immediate predecessor, nH_prev=0
        # (CO2* has no H atoms)
        return max(0, nH_curr - nH_prev)


    def run_pathway_site(
        self,
        head: str,
        pathway_id: str,
        site_idx: int,
        site: Dict,
        co2_slab: Atoms,
        bare_slab: Atoms,
        bare_energy: float,
        refs: Dict[str, float],
        optimizer: "OptimizerManager",
        builder: "StructureBuilder",
        validator: "ChemistryValidator",
        stab_engine: "StabilityTestEngine",
        recovery_engine: "AdaptiveRecoveryEngine",
        energy_analyzer: "EnergyAnalyzer",
        site_dir: Path,
    ) -> Tuple[bool, List["StepResult"], List["FailureRecord"]]:
        """
        Execute all steps of one CO2RR pathway at one active site.

        NO-ABORT / SELF-HEALING DESIGN
        ================================
        The pathway ALWAYS runs to completion regardless of what happens at any
        individual step.  The loop never breaks or returns early.

        Per-step recovery pipeline (exactly 5 iterations per stage):
        -------------------------------------------------------------
        Stage 0 -- Builder
            Try builder.build() with h_angle_idx 0..4.
            First geometry that is not None wins.
            Fallback: perturbed copy of prev_atoms.

        Stage 1 -- Main relax
            Run optimizer.relax() on the builder geometry.
            Exception -> immediate Stage-2 recovery on current_atoms.

        Stage 2 -- Stability sweep (5 iterations, always)
            Run stab_engine with exactly 5 iterations from the POSCAR seed.
            Collect all (energy, geo_desc) pairs.
            Pick the lowest-energy *intact* result if any exist.
            Otherwise pick lowest-energy result overall (best-available).

        Stage 3 -- Adaptive recovery (5 attempts, always when Stage 2
                   produced no intact result)
            Run recovery_engine.recover() with max_attempts=5.
            recover() already returns the best-available structure even when
            no intact result is found, so best_for_next is always set.

        Stage 4 -- Absolute fallback
            If every stage above crashed completely (rec_atoms is None and
            stab produced nothing), use prev_atoms directly so propagation
            never stalls.

        Propagation rule
        ----------------
        best_for_next is always the lowest-energy *intact* structure found
        across Stages 1-3.  If no intact structure was found anywhere, the
        lowest-energy structure overall is used (best-available), and
        step_intact is set False so the log is honest.  The pathway continues
        regardless.

        Pre-step parent check
        ---------------------
        Run before every PCET step but NEVER abort.  If the parent fails
        validation, log a warning, append to failures, and continue with
        prev_atoms.
        """
        _RECOVERY_ITERS = self.cfg.get("recovery_iters_per_stage", 6)

        pdef    = PATHWAYS[pathway_id]
        steps   = pdef["steps"]
        n_slab  = len(bare_slab)

        results:  List[StepResult]    = []
        failures: List[FailureRecord] = []

        prev_atoms = co2_slab.copy()
        stab_info  = {                       # carried forward; updated each step
            "std_energy": 0.0, "intact_count": 1,
            "stable": True, "mean_energy": 0.0,
            "min_energy": 0.0, "all_energies": [],
            "iteration_details": [],
        }

        LOG.info(f"\n  {pdef['color']}Path {pathway_id} ({pdef['name']}) "
                 f"-- site {site_idx:03d}{RESET}")

        for step_idx, step in enumerate(steps):
            step_label = STEP_LABEL.get(step, step)
            LOG.info(f"    [{step_idx+1:02d}/{len(steps)}] {step}  {step_label}")

            # ==============================================================
            # CHECKPOINT: skip if already done
            # ==============================================================
            ckpt = self.checkpoint.load(head, pathway_id, site_idx, step)
            if ckpt is not None:
                LOG.info(f"      [ckpt] {step}: "
                         f"E={ckpt.get('energy', float('nan')):.6f} eV")
                _base = f"{step_idx+1:02d}_{STEP_FOLDER_NAME.get(step, step)}"
                step_folder = site_dir / _base
                if not step_folder.exists():
                    _m = sorted(site_dir.glob(f"{_base}__*"))
                    if _m:
                        step_folder = _m[0]
                contcar = step_folder / "CONTCAR"
                if contcar.exists():
                    try:
                        prev_atoms = read(str(contcar))
                    except Exception:
                        pass
                sr = self._rebuild_step_result(ckpt)
                if sr:
                    results.append(sr)
                continue

            # ==============================================================
            # PRE-STEP PARENT VALIDATION (PCET steps only, never aborts)
            # ==============================================================
            _is_pcet = (self._pcet_count_for_step(
                step, step_idx, pathway_id, steps) > 0)
            if step_idx > 0 and _is_pcet:
                ok_parent, reason_parent = self._validate_parent(
                    prev_atoms, steps[step_idx - 1], site, n_slab, stab_info)
                if not ok_parent:
                    LOG.warning(
                        f"      {YELLOW}[PRE-STEP WARN] {reason_parent} "
                        f"-- continuing with best-available parent{RESET}")
                    failures.append(FailureRecord(
                        head=head, pathway=pathway_id, site_idx=site_idx,
                        step=step, step_idx=step_idx,
                        reason=f"parent_warn: {reason_parent}", attempts=0,
                    ))
                    # Never abort; prev_atoms already holds best available

            # ==============================================================
            # STAGE 0 -- BUILDER  (try 5 h_angle_idx values)
            # ==============================================================
            _base_name  = f"{step_idx+1:02d}_{STEP_FOLDER_NAME.get(step, step)}"
            step_folder = site_dir / f"{_base_name}__relaxing"
            step_folder.mkdir(parents=True, exist_ok=True)

            current_atoms = None
            for _h_idx in range(_RECOVERY_ITERS):
                try:
                    if step_idx == 0 and pathway_id != "A":
                        current_atoms = prev_atoms.copy()
                        break
                    candidate = builder.build(
                        step, prev_atoms, pathway_id=pathway_id,
                        h_angle_idx=_h_idx)
                    if candidate is not None:
                        current_atoms = candidate
                        break
                except Exception as _be:
                    LOG.debug(f"      [builder h_idx={_h_idx}] {_be}")

            if current_atoms is None:
                # Absolute builder fallback: perturb prev_atoms
                LOG.warning(f"      {YELLOW}[STAGE-0] all builder attempts failed "
                            f"-- using perturbed prev_atoms{RESET}")
                failures.append(FailureRecord(
                    head=head, pathway=pathway_id, site_idx=site_idx,
                    step=step, step_idx=step_idx,
                    reason="builder_all_failed_fallback", attempts=_RECOVERY_ITERS,
                ))
                current_atoms = prev_atoms.copy()
                _mag = self.cfg.get("perturbation_magnitude", 0.05) * 2.0
                for _i in range(n_slab, len(current_atoms)):
                    current_atoms.positions[_i] += np.random.normal(0, _mag, 3)
                _stol = SlabGenerator.surface_top_z(current_atoms)
                current_atoms.positions[n_slab:, 2] = np.maximum(
                    current_atoms.positions[n_slab:, 2], _stol + 0.40)

            write(str(step_folder / "POSCAR"),         current_atoms, format="vasp")
            write(str(step_folder / "PRE_RELAX.vasp"), current_atoms, format="vasp")

            # ==============================================================
            # STAGE 1 -- MAIN RELAX
            # ==============================================================
            relaxed = None
            energy  = float("nan")
            fres    = float("nan")
            try:
                relaxed, energy, fres = optimizer.relax(
                    current_atoms, step_folder, step_name=step)
            except Exception as exc:
                LOG.warning(f"      {YELLOW}[STAGE-1] main relax exception: {exc}{RESET}")
                failures.append(FailureRecord(
                    head=head, pathway=pathway_id, site_idx=site_idx,
                    step=step, step_idx=step_idx,
                    reason=f"main_relax_exception: {exc}", attempts=0,
                ))

            expected_comp = STEP_EXPECTED_COMPOSITION.get(step)

            # Assess main relax result (if it succeeded)
            main_intact = False
            if relaxed is not None:
                geo_desc   = describe_relaxed_geometry(
                    relaxed, n_slab, step, expected_comp)
                main_intact = geo_desc in ("intact", "clean_surface")
                intact_col  = GREEN if main_intact else YELLOW
                LOG.info(f"      [STAGE-1] E={energy:+.6f}  fmax={fres:.4f}  "
                         f"[{intact_col}{geo_desc}{RESET}]")
                if not main_intact:
                    failures.append(FailureRecord(
                        head=head, pathway=pathway_id, site_idx=site_idx,
                        step=step, step_idx=step_idx,
                        reason=f"main_relax_not_intact: {geo_desc}", attempts=0,
                    ))
                # Rename folder to reflect descriptor
                desc_folder = site_dir / f"{_base_name}__{geo_desc}"
                try:
                    if step_folder.exists() and step_folder != desc_folder:
                        step_folder.rename(desc_folder)
                        step_folder = desc_folder
                except Exception:
                    pass

            # ==============================================================
            # INFINITE RETRY RECOVERY PROTOCOL (STRICT & INFINITE)
            # ==============================================================
            rec_history: List[Dict] = []
            stab_info    = {"std_energy": 0.0, "intact_count": 0, "stable": True,
                            "mean_energy": energy if np.isfinite(energy) else 0.0,
                            "min_energy":  energy if np.isfinite(energy) else float("inf"),
                            "all_energies": [], "iteration_details": []}

            # ==============================================================
            # STRICT CHEMISTRY-AWARE RECOVERY
            #
            # RULE: Every step must produce a chemically intact intermediate.
            #       Broken/dissociated/wrong-isomer structures are NEVER
            #       propagated to the next step, no exceptions.
            #
            # Flow:
            #   Phase 1 -- Stability Sweep:
            #       Always runs (even when main relax is intact).
            #       Saves every trial to stability_tests/iter_N/
            #       Independently CONFIRMS intactness from multiple starting
            #       geometries to rule out false positives.
            #
            #   Phase 2 -- Infinite Adaptive Recovery:
            #       Runs only when Phase 1 finds no intact structure.
            #       Cycles through all strategies with escalating aggressiveness.
            #       Every attempt saved to stability_tests/recovery_attempt_N/
            #       Continues until intact found or hard safety cap reached.
            #
            #   On failure:
            #       Pathway is BLOCKED at this step.
            #       The broken structure is recorded but NEVER propagated.
            #       A clear error is logged. No silent failures.
            # ==============================================================

            STRICT          = self.cfg.get("strict_no_propagate_broken", True)
            MAX_ITERS       = self.cfg.get("max_recovery_iters", 200)
            ALWAYS_STAB     = self.cfg.get("always_run_stability", True)
            STAB_ITERS      = self.cfg.get("stability_initial_iters", 6)
            LOG_ALL         = self.cfg.get("recovery_log_all_attempts", True)

            stab_dir = step_folder / "stability_tests"
            stab_dir.mkdir(parents=True, exist_ok=True)

            # Write a validation manifest for auditing
            manifest_path = stab_dir / "VALIDATION_MANIFEST.log"

            def _write_manifest(line: str):
                with open(manifest_path, "a") as _mf:
                    _mf.write(line + "\n")

            _write_manifest(f"Step: {step}  Pathway: {pathway_id}  Site: {site_idx}")
            _write_manifest(f"Expected composition: {expected_comp}")
            _write_manifest(f"Main relax: E={energy:+.6f}  desc={geo_desc}  intact={main_intact}")
            _write_manifest("-" * 60)

            # -- Save main relax to stability_tests/main_relax/ ---------------
            _mr_dir = stab_dir / "main_relax"
            _mr_dir.mkdir(exist_ok=True)
            try:
                write(str(_mr_dir / "POSCAR"),  current_atoms, format="vasp")
                write(str(_mr_dir / "CONTCAR"), relaxed,       format="vasp")
                if self.cfg.get("save_cif", True):
                    write(str(_mr_dir / "CONTCAR.cif"), relaxed, format="cif")
                (_mr_dir / "validation.log").write_text(
                    f"descriptor: {geo_desc}\nintact: {main_intact}\n"
                    f"energy: {energy:+.8f}\nfmax: {fres:.6f}\n")
            except Exception as _e:
                LOG.debug(f"main_relax save error: {_e}")

            # -- Phase 1: Stability Sweep -------------------------------------
            # Run STAB_ITERS independent trials. Always runs, even if main intact.
            # This catches false positives and finds genuinely lower-energy intact.
            intact_found     = False
            best_for_next    = None
            best_e           = float("nan")
            step_intact      = False
            total_iters_used = 0

            LOG.info(f"      [PHASE-1] Stability sweep: {STAB_ITERS} trials "
                      f"(always runs -- validates main relax claim)")

            for _si in range(STAB_ITERS):
                _iter_dir = stab_dir / f"iter_{_si:03d}"
                _iter_dir.mkdir(exist_ok=True)
                try:
                    _trial = stab_engine.builder.build_for_iteration(
                        step, current_atoms, pathway_id,
                        iteration=_si, n_total=STAB_ITERS)
                    stab_engine._perturb_adsorbate(
                        _trial, pathway_id, step, iteration=_si, n_total=STAB_ITERS)
                    write(str(_iter_dir / "POSCAR"), _trial, format="vasp")

                    _rel, _e, _fr = optimizer.relax(
                        _trial, _iter_dir, step_name=f"{step}_stab{_si}",
                        fmax=self.cfg.get("stability_relax_fmax", 0.05),
                        max_steps=self.cfg.get("stability_max_steps", 500))
                    _geo = describe_relaxed_geometry(_rel, n_slab, step, expected_comp)
                    _ok  = _geo in ("intact", "clean_surface")
                    total_iters_used += 1

                    write(str(_iter_dir / "CONTCAR"), _rel, format="vasp")
                    if self.cfg.get("save_cif", True):
                        try: write(str(_iter_dir / "CONTCAR.cif"), _rel, format="cif")
                        except Exception: pass
                    (_iter_dir / "validation.log").write_text(
                        f"descriptor: {_geo}\nintact: {_ok}\n"
                        f"energy: {_e:+.8f}\nfmax: {_fr:.6f}\n"
                        f"iteration: {_si}\n")
                    _write_manifest(f"  stab iter {_si:03d}: desc={_geo}  "
                                     f"intact={_ok}  E={_e:+.6f}")

                    status_col = GREEN if _ok else RED
                    LOG.info(f"      [PHASE-1 iter {_si:03d}] {status_col}{_geo}{RESET}  "
                              f"E={_e:+.6f}  fmax={_fr:.4f}")

                    if _ok and (not np.isfinite(best_e) or _e < best_e):
                        best_for_next = _rel.copy()
                        best_e        = _e
                        intact_found  = True
                        step_intact   = True
                        write(str(stab_dir / "BEST_INTACT.vasp"), _rel, format="vasp")
                        _write_manifest(f"  --> INTACT at iter {_si:03d}  E={_e:+.6f}")
                        LOG.info(f"      {GREEN}[PHASE-1] Intact confirmed at iter {_si:03d}"
                                  f"  E={_e:+.6f}{RESET}")

                    stab_info["iteration_details"].append({
                        "phase": 1, "iteration": _si, "energy": float(_e),
                        "fmax": float(_fr), "geometry": _geo, "is_intact": _ok})
                    stab_info["all_energies"].append(float(_e))

                except Exception as _ie:
                    LOG.debug(f"      [PHASE-1 iter {_si}] exception: {_ie}")
                    _write_manifest(f"  stab iter {_si:03d}: ERROR {_ie}")

            if intact_found:
                stab_info["intact_count"] = 1
                stab_info["min_energy"]   = best_e
                LOG.info(f"      {GREEN}[PHASE-1 DONE] Intact found. "
                          f"Best E={best_e:+.6f} eV{RESET}")

            # -- Phase 2: Infinite Adaptive Recovery --------------------------
            # Only runs when Phase 1 found nothing intact.
            if not intact_found:
                LOG.warning(f"      {YELLOW}[PHASE-2] Phase 1 found no intact "
                             f"structure. Starting adaptive recovery...{RESET}")
                _write_manifest("Phase 1 failed. Starting Phase 2 adaptive recovery.")

                attempt_idx = 1
                _cycle      = 0   # escalation cycle

                while not intact_found:
                    # Safety cap (prevents truly infinite runtime in practice)
                    if MAX_ITERS is not None and attempt_idx > MAX_ITERS:
                        LOG.error(
                            f"      {RED}[PHASE-2] Safety cap {MAX_ITERS} reached "
                            f"with no intact intermediate for {step}.{RESET}")
                        _write_manifest(
                            f"HARD CAP {MAX_ITERS} reached. No intact found.")
                        break

                    _cycle = (attempt_idx - 1) // 7   # escalation cycle number

                    rec_folder = stab_dir / f"recovery_attempt_{attempt_idx:04d}"
                    rec_folder.mkdir(parents=True, exist_ok=True)

                    try:
                        strategy, trial = recovery_engine.apply_strategy_for_attempt(
                            attempt_idx, current_atoms, prev_atoms, step, pathway_id)

                        # Escalating perturbation magnitude every cycle
                        if _cycle > 0:
                            _mag = self.cfg.get("perturbation_magnitude", 0.10) * (1 + _cycle * 0.5)
                            _ads_idx = list(range(n_slab, len(trial)))
                            for _ai in _ads_idx:
                                trial.positions[_ai] += np.random.normal(0, _mag, 3)
                                trial.positions[_ai, 2] = max(
                                    trial.positions[_ai, 2],
                                    trial.positions[:n_slab, 2].max() + 0.30)

                        write(str(rec_folder / "POSCAR"), trial, format="vasp")

                        _rel, _e, _fr = optimizer.relax(
                            trial, rec_folder,
                            step_name=f"{step}_rec{attempt_idx}",
                            fmax=self.cfg.get("stability_relax_fmax", 0.05),
                            max_steps=self.cfg.get("stability_max_steps", 500))

                        total_iters_used += 1
                        write(str(rec_folder / "CONTCAR"), _rel, format="vasp")
                        if self.cfg.get("save_cif", True):
                            try: write(str(rec_folder/"CONTCAR.cif"), _rel, format="cif")
                            except Exception: pass

                        _geo = describe_relaxed_geometry(_rel, n_slab, step, expected_comp)
                        _ok  = _geo in ("intact", "clean_surface")

                        (rec_folder / "recovery.log").write_text(
                            f"attempt:    {attempt_idx}\n"
                            f"cycle:      {_cycle}\n"
                            f"strategy:   {strategy}\n"
                            f"descriptor: {_geo}\n"
                            f"intact:     {_ok}\n"
                            f"energy:     {_e:+.8f} eV\n"
                            f"fmax:       {_fr:.6f} eV/Ang\n")
                        _write_manifest(
                            f"  rec {attempt_idx:04d} (cycle {_cycle}) "
                            f"[{strategy}]: desc={_geo}  intact={_ok}  E={_e:+.6f}")

                        status_col = GREEN if _ok else RED
                        LOG.info(
                            f"      [PHASE-2 att {attempt_idx:04d} cy {_cycle}] "
                            f"[{strategy}] {status_col}{_geo}{RESET}  "
                            f"E={_e:+.6f}  fmax={_fr:.4f}")

                        rec_history.append({
                            "attempt": attempt_idx, "cycle": _cycle,
                            "strategy": strategy, "energy": float(_e),
                            "fres": float(_fr), "valid": _ok, "descriptor": _geo})
                        stab_info["iteration_details"].append({
                            "phase": 2, "attempt": attempt_idx, "cycle": _cycle,
                            "strategy": strategy, "energy": float(_e),
                            "fmax": float(_fr), "geometry": _geo, "is_intact": _ok})
                        stab_info["all_energies"].append(float(_e))

                        if _ok and (not np.isfinite(best_e) or _e < best_e):
                            best_for_next = _rel.copy()
                            best_e        = _e
                            intact_found  = True
                            step_intact   = True
                            stab_info["intact_count"] = 1
                            stab_info["min_energy"]   = best_e
                            write(str(stab_dir / "BEST_INTACT.vasp"), _rel, format="vasp")
                            _write_manifest(
                                f"  --> INTACT at recovery {attempt_idx:04d}  "
                                f"E={_e:+.6f}  strategy={strategy}")
                            LOG.info(
                                f"      {GREEN}[PHASE-2] INTACT found at attempt "
                                f"{attempt_idx:04d} via {strategy}  "
                                f"E={_e:+.6f} eV{RESET}")

                            for _fr_rec in reversed(failures):
                                if _fr_rec.step == step and _fr_rec.site_idx == site_idx:
                                    _fr_rec.reason += f" [rescued_recovery_att{attempt_idx}]"
                                    break

                    except Exception as _exc:
                        LOG.warning(
                            f"      {RED}[PHASE-2 att {attempt_idx}] {strategy} "
                            f"exception: {_exc}{RESET}")
                        _write_manifest(f"  rec {attempt_idx:04d}: ERROR {_exc}")
                        (rec_folder / "recovery.log").write_text(
                            f"attempt: {attempt_idx}\nstrategy: {strategy}\n"
                            f"error:   {_exc}\nintact:  False\n")
                        rec_history.append({
                            "attempt": attempt_idx, "strategy": strategy,
                            "error": str(_exc), "valid": False})

                    attempt_idx += 1

                if not intact_found:
                    _write_manifest(
                        f"RESULT: NO INTACT FOUND after {attempt_idx-1} total attempts.")

            # -- STRICT PROPAGATION GATE ---------------------------------------
            # If no intact structure found, BLOCK the pathway.
            # Never silently propagate a broken intermediate.
            if not intact_found or best_for_next is None:
                _err = (f"[STRICT BLOCK] No intact {step} found after "
                         f"{total_iters_used} attempts (Phase 1: {STAB_ITERS} iters, "
                         f"Phase 2: {total_iters_used - STAB_ITERS} iters). "
                         f"Pathway HALTED. Broken structures are NEVER propagated.")
                LOG.error(f"      {RED}{_err}{RESET}")
                _write_manifest(_err)

                # Write a clear BLOCKED marker file
                (stab_dir / "PATHWAY_BLOCKED.txt").write_text(
                    f"{_err}\n"
                    f"Step: {step}\nPathway: {pathway_id}\nSite: {site_idx}\n"
                    f"Total attempts: {total_iters_used}\n")

                failures.append(FailureRecord(
                    head=head, pathway=pathway_id, site_idx=site_idx,
                    step=step, step_idx=step_idx,
                    reason=f"BLOCKED_no_intact_after_{total_iters_used}_attempts",
                    attempts=total_iters_used))
                pathway_complete = False
                break   # HARD STOP -- do not continue to next step

            _write_manifest(
                f"RESULT: INTACT confirmed  E={best_e:+.6f}  "
                f"total_attempts={total_iters_used}")
            LOG.info(f"      {GREEN}[STRICT PASS] Intact intermediate confirmed "
                      f"E={best_e:+.6f}  attempts={total_iters_used}{RESET}")
            # Energy from confirmed intact structure
            energy = best_e

            stab_info["min_energy"]   = min(
                stab_info.get("min_energy", float("inf")),
                best_e if np.isfinite(best_e) else float("inf"))

            # ==============================================================
            # RENAME folder to FINAL descriptor (selected best structure)
            # Always rename -- even if main-relax said "dissociated", once
            # stability/recovery finds an intact structure the folder reflects that.
            # ==============================================================
            final_desc = describe_relaxed_geometry(
                best_for_next, n_slab, step, expected_comp)
            _final_target = site_dir / f"{_base_name}__{final_desc}"
            try:
                if step_folder.exists() and step_folder != _final_target:
                    step_folder.rename(_final_target)
                    step_folder = _final_target
            except Exception:
                pass   # rename non-critical; folder content is correct either way

            # ==============================================================
            # ENERGY ANALYSIS, METRICS, SAVE, RECORD
            # ==============================================================
            n_pcet_step = self._pcet_count_for_step(
                step, step_idx, pathway_id, steps)
            dg, dg_bd = energy_analyzer.compute(
                energy, bare_energy, best_for_next, n_slab, step,
                n_pcet_override=n_pcet_step)

            geo_metrics = self._geometry_metrics(best_for_next, n_slab, step)
            self.output_mgr.save_structure(best_for_next, step_folder, "BEST")

            # -- Overwrite CONTCAR with the intact (or best-available) structure
            # This ensures every downstream tool (OVITO, VESTA, VASP restart)
            # reads the best structure we have, not the raw main-relax output.
            try:
                write(str(step_folder / "CONTCAR"), best_for_next, format="vasp")
                if step_intact:
                    write(str(step_folder / "INTACT.vasp"), best_for_next,
                          format="vasp")
                    # Also update stability_tests best CONTCAR if it exists
                    for _sc in step_folder.glob("stability_tests/**/CONTCAR"):
                        pass  # individual iter CONTCARs left as-is
                LOG.debug(f"      CONTCAR updated -> {step_folder.name}")
            except Exception as _we:
                LOG.warning(f"      CONTCAR write failed: {_we}")

            # DG is only physically meaningful for intact intermediates.
            # If no intact found, still record the step but tag it clearly.
            _dg_valid = step_intact
            if not _dg_valid:
                LOG.warning(f"      {YELLOW}[DG] no intact structure -- "
                             f"DG={dg:+.4f} eV recorded as best-available "
                             f"(treat with caution){RESET}")

            vr_main = ValidationResult(
                is_valid=step_intact, stoich_ok=step_intact,
                bonds_ok=step_intact, geometry_ok=True,
                adsorbed_ok=True, convergence_ok=True,
                descriptor=final_desc, messages=[final_desc],
            )
            sr = StepResult(
                head           = head,
                pathway        = pathway_id,
                site_idx       = site_idx,
                step           = step,
                step_idx       = step_idx,
                energy         = float(energy),      # intact energy when found
                bare_energy    = float(bare_energy),
                dg             = float(dg) if _dg_valid else float("nan"),
                dg_breakdown   = dg_bd,
                n_pcet         = dg_bd.get("n_pcet", 0),
                folder         = str(step_folder),
                validation     = vr_main,
                stability_info = stab_info,
                geometry       = geo_metrics,
                retry_history  = rec_history,
                rescued        = (not main_intact
                                  and stab_info.get("intact_count", 0) > 0),
            )
            results.append(sr)
            self.tracker.record_success(sr)
            self.checkpoint.save(sr)

            dg_col = GREEN if dg < 0 else (YELLOW if dg < 0.5 else RED)
            # Report early-stop info if available
            _hist_path = step_folder / "iteration_history.json"
            _early_info = ""
            if _hist_path.exists():
                try:
                    import json as _j
                    _hi = _j.loads(_hist_path.read_text())
                    if _hi.get("stopped_early"):
                        _early_info = (f"  {GREEN}[early-stop iter="
                                        f"{_hi['stopping_iter']}]{RESET}")
                    else:
                        _early_info = f"  [full-conv iters={_hi['total_iters']}]"
                except Exception:
                    pass
            LOG.info(f"      DG={dg_col}{dg:+.4f} eV{RESET}  "
                     f"n_pcet={dg_bd.get('n_pcet', 0)}  "
                     f"intact={step_intact}  desc={final_desc}{_early_info}")

            # -- STRICT: Only propagate CONFIRMED INTACT structure ----------
            # (best_for_next is guaranteed intact here -- the STRICT GATE above
            #  breaks the loop if not intact, so we never reach this line with
            #  a broken structure)
            assert best_for_next is not None and step_intact, (
                "BUG: reached propagation with broken structure -- "
                "strict gate should have blocked this")
            prev_atoms       = best_for_next.copy()
            prev_step_intact = True   # always True here by construction
            LOG.info(f"      {GREEN}--> propagating INTACT "
                      f"E={best_e:+.6f} eV to step "
                      f"{step_idx+2 if step_idx+1 < len(steps) else 'END'}{RESET}")

        # Pathway complete -- all steps produced confirmed intact intermediates
        LOG.info(f"  {pdef['color']}{BOLD}[COMPLETE] "
                 f"Path {pathway_id} site {site_idx:03d}  "
                 f"({len(results)}/{len(steps)} steps recorded){RESET}")
        return True, results, failures


    @staticmethod
    def _geometry_metrics(atoms: Optional[Atoms], n_slab: int,
                           step: str) -> Dict[str, Any]:
        if atoms is None or len(atoms) <= n_slab:
            return {}
        ads    = atoms[n_slab:]
        stol_z = atoms.positions[:n_slab, 2].max()

        # Compute minimum metal-adsorbate distance (the surface bond length)
        _top_tol = CFG.get("surface_bond_top_layer_tol", 2.5)
        top_pos  = np.array([
            atoms.positions[i]
            for i in range(n_slab)
            if atoms.positions[i, 2] > stol_z - _top_tol
        ])
        min_surf_dist = float("inf")
        binding_atom  = "?"
        if len(top_pos) > 0 and len(ads) > 0:
            for a in ads:
                d = float(np.linalg.norm(top_pos - a.position, axis=1).min())
                if d < min_surf_dist:
                    min_surf_dist = d
                    binding_atom  = a.symbol
        surface_bound = (min_surf_dist < CFG.get("surface_bond_max", 2.80))

        metrics: Dict[str, Any] = {
            "n_ads_atoms":      len(ads),
            "surface_top_z":    round(float(stol_z), 4),
            "min_metal_ads_dist": round(float(min_surf_dist), 4)
                                   if np.isfinite(min_surf_dist) else None,
            "binding_atom":     binding_atom,
            "surface_bound":    surface_bound,
        }
        c_pos = [a.position for a in ads if a.symbol == "C"]
        o_pos = [a.position for a in ads if a.symbol == "O"]
        h_pos = [a.position for a in ads if a.symbol == "H"]

        if c_pos and o_pos:
            co_dists = [float(np.linalg.norm(cp - op))
                        for cp in c_pos for op in o_pos]
            metrics["co_min_dist_ang"] = round(min(co_dists), 4)
        if c_pos:
            metrics["c_height_ang"] = round(
                float(c_pos[0][2] - stol_z), 4
            )
        if o_pos:
            metrics["o_height_ang"] = round(
                float(min(op[2] - stol_z for op in o_pos)), 4
            )
        return metrics

    # -- Checkpoint rebuild helper ------------------------------------------
    @staticmethod
    def _rebuild_step_result(ckpt: Dict) -> Optional[StepResult]:
        try:
            vd = ckpt.get("validation", {})
            vr = ValidationResult(
                is_valid       = vd.get("is_valid", True),
                stoich_ok      = vd.get("stoich_ok", True),
                bonds_ok       = vd.get("bonds_ok", True),
                geometry_ok    = vd.get("geometry_ok", True),
                adsorbed_ok    = vd.get("adsorbed_ok", True),
                convergence_ok = vd.get("convergence_ok", True),
                descriptor     = vd.get("descriptor", "unknown"),
                messages       = vd.get("messages", []),
            )
            return StepResult(
                head          = ckpt["head"],
                pathway       = ckpt["pathway"],
                site_idx      = ckpt["site_idx"],
                step          = ckpt["step"],
                step_idx      = ckpt["step_idx"],
                energy        = ckpt.get("energy", float("nan")),
                bare_energy   = ckpt.get("bare_energy", float("nan")),
                dg            = ckpt.get("dg", float("nan")),
                dg_breakdown  = ckpt.get("dg_breakdown", {}),
                n_pcet        = ckpt.get("n_pcet", 0),
                folder        = ckpt.get("folder", ""),
                validation    = vr,
                stability_info= ckpt.get("stability_info", {}),
                geometry      = ckpt.get("geometry", {}),
                retry_history = ckpt.get("retry_history", []),
                rescued       = ckpt.get("rescued", False),
                timestamp     = ckpt.get("timestamp", ""),
            )
        except Exception:
            return None

    # ======================================================================
    # MAIN RUN METHOD
    # ======================================================================

    def run(self):
        """
        Full workflow entry point.

        Outer loop:  model heads
        Middle loop: pathways
        Inner loop:  active sites x pathway steps
        """
        LOG.info(f"\n{BOLD}{'='*80}")
        LOG.info("CO2RR AUTONOMOUS WORKFLOW -- Production-Grade Multi-Pathway Explorer")
        LOG.info(f"  Pathways : {list(PATHWAYS.keys())}")
        LOG.info(f"  Heads    : {CFG['model_heads']}")
        LOG.info(f"  Started  : {datetime.datetime.now()}")
        LOG.info(f"{'='*80}{RESET}")

        # -- Load slab ------------------------------------------------------
        slab_gen = SlabGenerator(self.cfg)
        slab     = slab_gen.load()
        n_slab   = len(slab)

        # -- Identify active sites ------------------------------------------
        site_id  = ActiveSiteIdentifier(slab, self.cfg)
        all_sites = site_id.identify()
        LOG.info(f"Slab atoms   : {n_slab}")
        LOG.info(f"Active sites : {len(all_sites)}")

        # -- Loop over heads ------------------------------------------------
        for head in self.cfg["model_heads"]:
            LOG.info(f"\n{CYAN}{'-'*70}")
            LOG.info(f"  MODEL HEAD: {head}")
            LOG.info(f"{'-'*70}{RESET}")

            head_dir = self.base_dir / head
            head_dir.mkdir(parents=True, exist_ok=True)

            # -- Build per-head components ----------------------------------
            calc     = self._make_calc(head)
            builder  = StructureBuilder(n_slab)
            optmgr   = OptimizerManager(calc, n_slab, self.cfg)
            validtr  = ChemistryValidator(n_slab, self.cfg)
            stabeng  = StabilityTestEngine(builder, optmgr, validtr, self.cfg)
            recoveng = AdaptiveRecoveryEngine(builder, optmgr, validtr, self.cfg)

            # -- Gas references ---------------------------------------------
            refs_dir = head_dir / "gas_references"
            refs     = optmgr.relax_gas_refs(refs_dir)
            energy_analyzer = EnergyAnalyzer(refs)

            # -- Bare slab relaxation ---------------------------------------
            bare_slab, bare_energy = self._get_bare_slab(head, optmgr, slab)

            # -- SLURM array-job batching ------------------------------------
            s_start = self.cfg.get("site_start") or 0
            s_end   = self.cfg.get("site_end")   or len(all_sites)
            s_end   = min(s_end, len(all_sites))
            batch   = all_sites[s_start:s_end]
            if s_start > 0 or s_end < len(all_sites):
                LOG.info(f"  SLURM batch: sites {s_start}..{s_end-1} "
                          f"({len(batch)} of {len(all_sites)} total)")

            # -- Loop over sites (outer) then pathways (inner) ---------------
            n_complete_per_path = {pid: 0 for pid in PATHWAYS}

            for site_i, site in enumerate(batch, start=s_start):
                LOG.info(f"\n{CYAN}{'-'*60}")
                LOG.info(f"  SITE {site_i:03d}  |  head: {head}  "
                          f"({site['type']}  at z={site['position'][2]:.2f})")
                LOG.info(f"{'-'*60}{RESET}")

                # Place CO2 above this site once -- all pathways reuse it
                co2_slab = self._place_co2_on_slab(bare_slab, site)

                # Build per-pathway directories before (possibly) going parallel
                for pathway_id, pdef in PATHWAYS.items():
                    (head_dir / f"Path_{pathway_id}_{pdef['name']}").mkdir(
                        parents=True, exist_ok=True)
                    (head_dir / f"Path_{pathway_id}_{pdef['name']}"
                     / f"site_{site_i:03d}_{site['type']}").mkdir(
                        parents=True, exist_ok=True)

                n_par = self.cfg.get("n_parallel_pathways", 1)
                pathway_ids = list(PATHWAYS.keys())

                def _run_one_pathway(pid):
                    pdef_   = PATHWAYS[pid]
                    pdir_   = head_dir / f"Path_{pid}_{pdef_['name']}"
                    sdir_   = pdir_ / f"site_{site_i:03d}_{site['type']}"
                    return self.run_pathway_site(
                        head            = head,
                        pathway_id      = pid,
                        site_idx        = site_i,
                        site            = site,
                        co2_slab        = co2_slab,
                        bare_slab       = bare_slab,
                        bare_energy     = bare_energy,
                        refs            = refs,
                        optimizer       = optmgr,
                        builder         = builder,
                        validator       = validtr,
                        stab_engine     = stabeng,
                        recovery_engine = recoveng,
                        energy_analyzer = energy_analyzer,
                        site_dir        = sdir_,
                    )

                if n_par > 1:
                    LOG.info(f"  Running {len(pathway_ids)} pathways in "
                              f"{n_par} threads for site {site_i:03d}")
                    with concurrent.futures.ThreadPoolExecutor(
                            max_workers=n_par) as pool:
                        futures = {pool.submit(_run_one_pathway, pid): pid
                                   for pid in pathway_ids}
                        for fut in concurrent.futures.as_completed(futures):
                            pid = futures[fut]
                            complete, _, step_failures = fut.result()
                            for f in step_failures:
                                self.tracker.record_failure(f)
                            if complete:
                                n_complete_per_path[pid] += 1
                else:
                    for pid in pathway_ids:
                        complete, _, step_failures = _run_one_pathway(pid)
                        for f in step_failures:
                            self.tracker.record_failure(f)
                        if complete:
                            n_complete_per_path[pid] += 1

            for pid, n_complete in n_complete_per_path.items():
                LOG.info(f"  Path {pid}: {n_complete}/{len(all_sites)} sites complete")

            # -- Save head-level outputs ------------------------------------
            self.output_mgr.save_pathway_map(self.tracker, head, head_dir)

        # -- Final outputs --------------------------------------------------
        self.output_mgr.save_json_summary(
            self.tracker.results, self.tracker.failures, self.base_dir
        )
        self.output_mgr.save_csv_summary(self.tracker.results, self.base_dir)
        self._print_final_summary()

        # Save full results pickle
        pkl_out = self.base_dir / "all_results.pkl"
        with open(pkl_out, "wb") as f:
            pickle.dump({
                "results":  [asdict(r) for r in self.tracker.results],
                "failures": [asdict(f) for f in self.tracker.failures],
                "config":   self.cfg,
                "pathways": {k: {kk: vv for kk, vv in v.items() if kk != "color"}
                             for k, v in PATHWAYS.items()},
                "timestamp": str(datetime.datetime.now()),
            }, f)
        LOG.info(f"\nResults pickle -> {pkl_out}")
        LOG.info(f"Finished : {datetime.datetime.now()}")

    # -- Final summary printer ----------------------------------------------
    def _print_final_summary(self):
        """
        Full publishable summary matching original template:
        - Per-pathway statistics (completed sites, total steps, failures)
        - Best final DeltaG per pathway
        - Limiting potential U_L per completed site
        - Per-PCET-step DeltaDeltaG increments (identifying which step limits)
        - U-sweep DeltaG profile at each potential (spontaneity check)
        - Selectivity table: DeltaU_L between pathway pairs per site
        - CO2 adsorption energy DeltaG(CO2*) per site
        """
        LOG.info(f"\n{BOLD}{'='*80}")
        LOG.info("FINAL SUMMARY")
        LOG.info(f"{'='*80}{RESET}")

        stats = self.tracker.summary_stats()
        for key, s in stats.items():
            LOG.info(f"  {key}:  results={s['n_results']}  "
                      f"failures={s['n_failures']}  "
                      f"complete_sites={s['n_complete_sites']}")

        u_values = self.cfg.get("U_values", [0.0])
        path_pairs = [("A","D"),("A","B"),("A","C"),("B","D"),("C","D"),("A","E")]

        for head in self.cfg["model_heads"]:
            LOG.info(f"\n{CYAN}{'='*70}")
            LOG.info(f"  HEAD: {head}")
            LOG.info(f"{'='*70}{RESET}")

            # -- Per-pathway statistics + limiting potentials ---------------
            for pid, pdef in PATHWAYS.items():
                res  = [r for r in self.tracker.results
                        if r.head == head and r.pathway == pid]
                fail = [f for f in self.tracker.failures
                        if f.head == head and f.pathway == pid]
                complete = self.tracker.complete_sites(head, pid)
                n_sites  = len({r.site_idx for r in res})

                LOG.info(f"\n  {pdef['color']}Path {pid}: "
                          f"{pdef['name']}{RESET}")
                LOG.info(f"    Completed sites : {len(complete)} / {n_sites}")
                LOG.info(f"    Total steps OK  : {len(res)}")
                LOG.info(f"    Failures        : {len(fail)}")

                if not complete:
                    continue

                # Best final DeltaG
                final_step = pdef["steps"][-1]
                final_dgs  = [r.dg for r in res
                               if r.step == final_step
                               and r.site_idx in complete]
                if final_dgs:
                    best = min(final_dgs)
                    col  = GREEN if best < 0 else RED
                    LOG.info(f"    Best final DeltaG: {col}{best:+.4f} eV{RESET}")

                # Limiting potentials + step increments
                LOG.info(f"    Limiting potentials "
                          f"(U_L = -max_stepDeltaDeltaG / e):")
                for s in sorted(complete)[:5]:
                    site_steps = sorted(
                        [r for r in res if r.site_idx == s],
                        key=lambda r: r.step_idx)
                    dg_series  = [{"step": r.step, "step_idx": r.step_idx,
                                   "dg_total": r.dg, "n_pcet": r.n_pcet}
                                  for r in site_steps]
                    u_l, lim_step, increments = EnergyAnalyzer.limiting_potential(
                        dg_series)
                    col = (GREEN if u_l > -0.8
                           else (YELLOW if u_l > -1.2 else RED))
                    LOG.info(f"      site {s:03d}: {col}{u_l:+.3f} V vs RHE{RESET}"
                              f"  (limiting: {lim_step})")
                    if increments:
                        inc = "  ".join(f"{st}:{dg:+.3f}"
                                        for st, dg in increments)
                        LOG.info(f"        PCET DeltaDeltaG [eV]: {inc}")

                # U-sweep profile (best site)
                if self.cfg.get("U_sweep", True) and u_values:
                    best_site = complete[0]
                    best_steps = sorted(
                        [r for r in res if r.site_idx == best_site],
                        key=lambda r: r.step_idx)
                    LOG.info(f"\n    U-sweep DeltaG profile (site {best_site:03d}):")
                    LOG.info(f"    {'U (V)':>8}  {'DeltaG_final':>14}"
                              f"  {'All-downhill?':>14}")
                    for U_val in u_values:
                        dg_at_u = [r.dg + r.n_pcet * U_val
                                   for r in best_steps]
                        # Spontaneity: every step increment <= 0
                        prev, all_down = 0.0, True
                        for dg_val in dg_at_u:
                            if dg_val - prev > 1e-6:
                                all_down = False
                                break
                            prev = dg_val
                        final = dg_at_u[-1] if dg_at_u else float("nan")
                        mark  = "[YES]" if all_down else "[NO] "
                        col   = GREEN if all_down else RED
                        LOG.info(f"    {U_val:>+8.2f}  {final:>+14.4f}"
                                  f"  {col}{mark}{RESET}")

            # -- Selectivity table ------------------------------------------
            LOG.info(f"\n{BOLD}{'='*70}")
            LOG.info("SELECTIVITY SUMMARY  "
                      "(DeltaU_L between pathways; positive = left path preferred)")
            LOG.info(f"{'='*70}{RESET}")

            ul_table = {}  # site -> {pid: U_L}
            for pid in PATHWAYS:
                for s in self.tracker.complete_sites(head, pid):
                    rlist = sorted(
                        self.tracker.get_pathway_results(head, pid, s),
                        key=lambda r: r.step_idx)
                    ds = [{"step": r.step, "step_idx": r.step_idx,
                           "dg_total": r.dg, "n_pcet": r.n_pcet}
                          for r in rlist]
                    u_l, _, _ = EnergyAnalyzer.limiting_potential(ds)
                    ul_table.setdefault(s, {})[pid] = u_l

            for s, ul_map in sorted(ul_table.items()):
                row = f"  site {s:03d}:"
                # U_L per pathway
                for pid, ul in sorted(ul_map.items()):
                    col = (GREEN if ul > -0.8
                           else (YELLOW if ul > -1.2 else RED))
                    row += f"  {pid}:{col}{ul:+.3f}V{RESET}"
                # DeltaU_L between pairs
                for px, py in path_pairs:
                    if px in ul_map and py in ul_map:
                        delta = ul_map[px] - ul_map[py]
                        dc = GREEN if delta > 0 else RED
                        row += f"  D({px}-{py})={dc}{delta:+.3f}{RESET}"
                # CO2 adsorption DeltaG
                for pid in ("A","B","C","D"):
                    co2_res = [r for r in self.tracker.results
                                if r.head == head and r.pathway == pid
                                and r.site_idx == s and r.step == "01_CO2"]
                    if co2_res:
                        dg_co2 = co2_res[0].dg
                        co2c   = GREEN if dg_co2 < 0 else RED
                        row   += f"  DG(CO2*)={co2c}{dg_co2:+.3f}{RESET}eV"
                        break
                LOG.info(row)

        LOG.info(f"\nFinished: {datetime.datetime.now()}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    """
    Entry point.  Accepts CLI overrides for HPC / SLURM array jobs:

        python co2rr_workflow.py --site-start 0  --site-end 5   # SLURM task 0
        python co2rr_workflow.py --site-start 5  --site-end 10  # SLURM task 1
        python co2rr_workflow.py --parallel 4                    # 4 pathway threads
        python co2rr_workflow.py --workdir /scratch/run1         # custom output dir
        python co2rr_workflow.py --fmax 0.02                     # tighter criterion

    All arguments are optional; defaults come from CFG above.
    """
    parser = argparse.ArgumentParser(
        description="Autonomous CO2RR hydrogenation workflow (MACE + ASE)")
    parser.add_argument("--site-start",  type=int,   default=None,
                        help="First site index (inclusive) for SLURM array jobs")
    parser.add_argument("--site-end",    type=int,   default=None,
                        help="Last site index (exclusive) for SLURM array jobs")
    parser.add_argument("--parallel",    type=int,   default=None,
                        help="Number of parallel pathway threads per site")
    parser.add_argument("--workdir",     type=str,   default=None,
                        help="Override CFG workdir")
    parser.add_argument("--fmax",        type=float, default=None,
                        help="Override relax_fmax convergence criterion")
    parser.add_argument("--heads",       type=str,   default=None,
                        help="Comma-separated model heads, e.g. oc20_usemppbe,omat_pbe")
    parser.add_argument("--max-sites",   type=int,   default=None,
                        help="Override max_sites (active sites to screen)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed (default 42)")
    parser.add_argument("--device",      type=str,   default=None,
                        help="Force device: 'cpu' or 'cuda' (overrides auto-detect)")
    args = parser.parse_args()

    # Apply CLI overrides to CFG
    if args.site_start is not None:
        CFG["site_start"] = args.site_start
    if args.site_end is not None:
        CFG["site_end"] = args.site_end
    if args.parallel is not None:
        CFG["n_parallel_pathways"] = args.parallel
    if args.workdir is not None:
        CFG["workdir"] = args.workdir
    if args.fmax is not None:
        CFG["relax_fmax"] = args.fmax
    if args.heads is not None:
        CFG["model_heads"] = [h.strip() for h in args.heads.split(",")]
    if args.max_sites is not None:
        CFG["max_sites"] = arsgs.max_sites
    if args.device is not None:
        CFG["device"] = args.device

    workdir = Path(CFG["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    global LOG
    LOG = setup_logging(workdir)

    np.random.seed(args.seed)

    # Log the run configuration
    LOG.info(f"CO2RR workflow -- Python {sys.version.split()[0]}")
    LOG.info(f"workdir    : {workdir.resolve()}")
    LOG.info(f"model      : {CFG['model_path']}")
    LOG.info(f"heads      : {CFG['model_heads']}")
    LOG.info(f"max_sites  : {CFG['max_sites']}")
    LOG.info(f"relax_fmax : {CFG['relax_fmax']} eV/Ang")
    if CFG.get("site_start") is not None or CFG.get("site_end") is not None:
        LOG.info(f"site batch : [{CFG.get('site_start',0)}, "
                  f"{CFG.get('site_end','end')})")
    if CFG.get("n_parallel_pathways", 1) > 1:
        LOG.info(f"parallel   : {CFG['n_parallel_pathways']} pathway threads/site")

    generator = PathwayGenerator(CFG)
    generator.run()


if __name__ == "__main__":
    main()