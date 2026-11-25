"""Compartment layout and dose-event kinds.

The index constants are module-level integers so numba kernels can use them as
compile-time constants.  Every age bin carries the same 39 compartments plus
three cumulative accumulators that are integrated with the state but never
shifted between bins.
"""

from __future__ import annotations

COMPARTMENTS: tuple[str, ...] = (
    "M",    # maternally protected, no infant dose
    "MV1",  # maternally protected + 1 infant dose
    "MV2",  # maternally protected + 2 infant doses
    "S0",   # never infected, never vaccinated
    "D1",   # 1 primary dose, no maternal protection
    "D2",   # 2 primary doses
    "Pw1",  # wP-primed protection, stage 1
    "Pw2",  # wP-primed protection, stage 2
    "Pa1",  # aP-primed protection (primary or child booster), stage 1
    "Pa2",
    "T1",   # Tdap booster protection in aP-primed people, stage 1
    "T2",
    "Sw",   # waned, Th1-primed (wP vaccination or past infection)
    "Sa",   # waned, aP-primed
    "Et",   # latent, typical course
    "Em",   # latent, mild course
    "It",   # infectious, typical
    "Im",   # infectious, mild
    "R",    # natural immunity
    "W",    # waned natural immunity, boostable
    "M_stage2", "MV1_stage2", "MV2_stage2",
    "M_stage3", "MV1_stage3", "MV2_stage3",
    "M_stage4", "MV1_stage4", "MV2_stage4",
    "Tw1", "Tw2",  # Tdap-only protection retaining wP priming
    "Ap11", "Ap12", "Ap21", "Ap22",  # primary aP stage x Tdap stage
    "Wp11", "Wp12", "Wp21", "Wp22",  # primary wP stage x Tdap stage
)
NC = len(COMPARTMENTS)
MODEL_VERSION = "overlapping_tdap_v2_39"
TW1, TW2 = 29, 30
AP11, AP12, AP21, AP22 = range(31, 35)
WP11, WP12, WP21, WP22 = range(35, 39)
JOINT_PROTECTION = ((AP11, AP12, AP21, AP22), (WP11, WP12, WP21, WP22))
PRIMARY_PROTECTION = ((8, 9), (6, 7))
TDAP_PROTECTION = ((10, 11), (TW1, TW2))

(M, MV1, MV2, S0, D1, D2, PW1, PW2, PA1, PA2, T1, T2, SW, SA, ET, EM, IT, IM, R, W) = range(20)

# States that can be infected (relative susceptibility sigma > 0 by default).
MATERNAL_STAGES = ((M, MV1, MV2), (20, 21, 22), (23, 24, 25), (26, 27, 28))
SUSCEPTIBLE_STATES: tuple[int, ...] = (M, MV1, MV2, S0, D1, D2, PW1, PW2, PA1, PA2, T1, T2, SW, SA) + tuple(range(20, NC))

ACCUMULATORS: tuple[str, ...] = ("onset_typical", "onset_mild", "infections")
NA = len(ACCUMULATORS)
ACC_TYPICAL, ACC_MILD, ACC_INFECTIONS = range(NA)

# Dose-event kinds.  Routing is by the recipient's current state; see
# ``kernel.apply_dose``.
DOSE1 = 0           # S0 -> D1, M -> MV1
DOSE2 = 1           # D1 -> D2, MV1 -> MV2
DOSE3 = 2           # D2/MV2 -> primary product stage 1 (completes primary series)
CHILD_BOOSTER = 3   # primed -> stage 1 of priming type; incomplete series advance
TDAP_BOOSTER = 4    # independent Tdap protection retaining primary clocks and priming
LATE_PRIMARY = 5    # unvaccinated or incompletely vaccinated -> primary product stage 1
N_DOSE_KINDS = 6
DOSE_KIND_NAMES: tuple[str, ...] = (
    "dose1", "dose2", "dose3", "child_booster", "tdap_booster", "late_primary",
)

# Infant-series dose tables cover dose1, dose2, dose3 and a toddler booster.
N_INFANT_DOSE_KINDS = 4

PRODUCT_WP = 0
PRODUCT_AP = 1
