import os

# ── PATHS ──────────────────────────────────────────────────────────────
# Default matches the plan doc (Google Colab + Drive). You can override via:
#   export GUARDRAILS_SYNTH_BASE="..."
BASE = os.environ.get(
    "GUARDRAILS_SYNTH_BASE",
    "/content/drive/MyDrive/guardrails-synthetic-data",
)

CHUNKS_DIR = f"{BASE}/data/chunks"
SYNTHETIC_DIR = f"{BASE}/data/synthetic"
EXPORTS_DIR = f"{BASE}/data/exports"

# ── CHUNK FILTERING THRESHOLDS ─────────────────────────────────────────
MIN_TOKENS = 80  # drop chunks below this
MAX_TOKENS = 420  # drop chunks above this (already enforced upstream per plan)
QUALITY_SCORE_MIN = 0.6  # minimum quality score to keep chunk (used in validation)

# ── GENERATION SETTINGS ────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS_RESPONSE = 2000
TEMPERATURE = 0.7
BATCH_SIZE = 10  # chunks per "batch" (loop grouping / checkpoint cadence)
RETRY_LIMIT = 3  # retries on API failure

# ── DATASET TARGETS ────────────────────────────────────────────────────
EMBEDDING_TARGET = 2500  # total triplets for Pipeline 1
AGENT_TARGET = 3000  # total pairs for Pipeline 2

# ── TRAIN/VAL/TEST SPLITS ──────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO = 0.12
TEST_RATIO = 0.08

# ── THREAT CATEGORIES (9 classes) ──────────────────────────────────────
THREAT_CATEGORIES = [
    "unauthorized_data_access",
    "pii_exfiltration",
    "prompt_injection",
    "policy_bypass",
    "sensitive_data_exposure",
    "cross_border_data_transfer",
    "model_extraction",
    "benign_allowed",
    "ambiguous_escalate",
]

# ── DECISION LABELS ────────────────────────────────────────────────────
DECISION_LABELS = ["BLOCK", "ALLOW", "ESCALATE"]

# ── TARGET LABEL DISTRIBUTION (Pipeline 2) ─────────────────────────────
LABEL_DISTRIBUTION = {
    "BLOCK": 0.50,
    "ALLOW": 0.30,
    "ESCALATE": 0.20,
}

