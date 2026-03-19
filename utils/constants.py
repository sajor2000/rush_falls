"""Column mappings, thresholds, labels, and project-wide constants."""

# ── Raw → Standardized column name mapping ──────────────────────────
COLUMN_RENAME_MAP: dict[str, str] = {
    "MRN": "mrn",
    "EncounterEpicCsn": "encounter_csn",
    "age_at_admission": "age",
    "Ethnicity": "ethnicity",
    "race": "race",
    "Gender": "gender",
    "accommodation_code": "accommodation_code",
    "admitting_department": "admitting_department",
    "discharge_department": "discharge_department",
    "Unit_fall_occurred": "unit_fall_occurred",
    "admission_date": "admission_date",
    "discharge_date": "discharge_date",
    "Primary_diagnosis": "primary_diagnosis",
    "FallDateTime": "fall_datetime",
    "Epic_score_before_fall": "epic_score_before_fall",
    "Morse_score_before_fall": "morse_score_before_fall",
    "epic_score_at_admission": "epic_score_admission",
    "morse_score_at_admission": "morse_score_admission",
    "Epic_score_max": "epic_score_max",
    "Epic_score_min": "epic_score_min",
    "Epic_score_mean": "epic_score_mean",
    "Epic_score_median": "epic_score_median",
    "morse_max": "morse_score_max",
    "morse_min": "morse_score_min",
    "morse_mean": "morse_score_mean",
    "morse_median": "morse_score_median",
}

# ── Accommodation codes classified as inpatient ─────────────────────
INPATIENT_CODES: set[str] = {
    "Inpatient",
    "Inpatient-Admit PRIOR To Surgery",
    "Inpatient-Admit After Surgery",
}

# ── Label cleaning maps (for figures / tables) ─────────────────────
RACE_CLEAN_MAP: dict[str, str] = {
    "White": "White",
    "White or Caucasian": "White",
    "Caucasian": "White",
    "Black": "Black",
    "Black or African American": "Black",
    "African American": "Black",
    "Asian": "Asian",
    "Native Hawaiian or Other Pacific Islander": "Asian",
    "Pacific Islander": "Asian",
    "American Indian or Alaska Native": "Other/Unknown",
    "American Indian": "Other/Unknown",
    "Other": "Other/Unknown",
    "Two or More Races": "Other/Unknown",
    "Multiracial": "Other/Unknown",
    "Unknown": "Other/Unknown",
    "Declined": "Other/Unknown",
    "Patient Declined": "Other/Unknown",
    "Unavailable": "Other/Unknown",
    "*Unspecified": "Other/Unknown",
}

ETHNICITY_CLEAN_MAP: dict[str, str] = {
    "Hispanic or Latino": "Hispanic/Latino",
    "Hispanic": "Hispanic/Latino",
    "Latino": "Hispanic/Latino",
    "Hispanic/Latino": "Hispanic/Latino",
    "Not Hispanic or Latino": "Not Hispanic/Latino",
    "Not Hispanic": "Not Hispanic/Latino",
    "Not Hispanic/Latino": "Not Hispanic/Latino",
    "Non-Hispanic": "Not Hispanic/Latino",
    "Unknown": "Unknown",
    "Declined": "Unknown",
    "Patient Declined": "Unknown",
    "Unavailable": "Unknown",
    "*Unspecified": "Unknown",
}

GENDER_CLEAN_MAP: dict[str, str] = {
    "Female": "Female",
    "Male": "Male",
    "female": "Female",
    "male": "Male",
    "F": "Female",
    "M": "Male",
    "Nonbinary": "Other",
    "Non-binary": "Other",
    "X": "Other",
    "Other": "Other",
    "Unknown": "Unknown",
    "Declined": "Unknown",
}

CATEGORICAL_COLS: list[str] = [
    "gender",
    "race",
    "ethnicity",
    "accommodation_code",
    "admitting_department",
    "discharge_department",
    "unit_fall_occurred",
]

# ── Department → unit type mapping ──────────────────────────────────
# Populated once admitting_department values are profiled in 01_data_discovery.
# Placeholder structure — fill with actual department names.
UNIT_TYPE_MAP: dict[str, str] = {
    # "Medicine 7N": "Medical",
    # "Surgical ICU": "ICU",
    # "Ortho 5S": "Surgical",
    # ...
}

# Default fallback when department not in map
UNIT_TYPE_DEFAULT = "Other"

# ── Score columns by timing strategy ────────────────────────────────
SCORE_TIMING = {
    "admission": {
        "epic": "epic_score_admission",
        "morse": "morse_score_admission",
    },
    "before_fall": {
        "epic": "epic_score_before_fall",
        "morse": "morse_score_before_fall",
    },
    "max": {
        "epic": "epic_score_max",
        "morse": "morse_score_max",
    },
    "mean": {
        "epic": "epic_score_mean",
        "morse": "morse_score_mean",
    },
}

# ── Standard Morse Fall Scale cutoffs ───────────────────────────────
MFS_LOW = 0       # Low risk: 0-24
MFS_MODERATE = 25  # Moderate risk: 25-44
MFS_HIGH = 45     # High risk: ≥45

# ── Epic PMFRS recommended cutoffs (from model brief, Aug 2025) ────
# WARNING: These thresholds are designed for continuous monitoring (max
# score over encounter), NOT admission-time screening.  At admission,
# 97% of encounters score < 35 in our dataset.
EPIC_3TIER_MEDIUM = 35   # 3-tier: Low 0-34, Medium 35-69, High 70-100
EPIC_3TIER_HIGH = 70
EPIC_2TIER_HIGH = 50     # 2-tier: Low 0-49, High 50-100

# ── Age group bins for fairness audit ───────────────────────────────
AGE_BINS = [18, 65, 80, float("inf")]
AGE_LABELS = ["18-64", "65-79", "≥80"]

# ── Bootstrap / statistical parameters ──────────────────────────────
N_BOOTSTRAP = 2000
RANDOM_SEED = 42
ALPHA = 0.05
MIN_SUBGROUP_EVENTS = 20

# ── Value-optimizing threshold (Parsons et al. JAMIA 2023) ─────────
COST_PARAMS = {
    "cost_fall_mean": 14_000,
    "cost_fall_sd": 3_000,
    "cost_intervention_mean": 200,
    "cost_intervention_sd": 50,
    "effectiveness_alpha": 40,   # Beta(40, 60) ≈ 0.4 mean
    "effectiveness_beta": 60,
    "qaly_loss_mean": 0.0036,
    "qaly_loss_sd": 0.0005,
    "wtp": 100_000,
    "n_mc": 1_000,
}

# ── DCA threshold range ─────────────────────────────────────────────
DCA_THRESHOLD_MIN = 0.0
DCA_THRESHOLD_MAX = 0.10
DCA_THRESHOLD_STEP = 0.001

# ── Datetime columns to parse ───────────────────────────────────────
DATETIME_COLS = ["admission_date", "discharge_date", "fall_datetime"]

# ── Model display labels ────────────────────────────────────────────
MODEL_LABELS = {
    "epic": "Epic PMFRS",
    "morse": "Morse Fall Scale",
}
