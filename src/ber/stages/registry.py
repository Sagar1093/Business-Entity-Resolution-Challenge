"""Stage registry: dependency order + metadata. Extended as stages are implemented."""
from . import s0_env, s1_validate, s2_normalize, s3_blocking, s_eda, s_splits

STAGES = {
    "s0_env": {"run": s0_env.run, "desc": "hardware/environment detection gate (S0)"},
    "validate_data": {"run": s1_validate.run, "desc": "S1 data validation gate over all 7 TSVs"},
    "eda": {"run": s_eda.run, "desc": "train-only noise statistics (reports/eda.md)"},
    "splits": {"run": s_splits.run, "desc": "grouped 90/10 split by S1 entity"},
    "normalize": {"run": s2_normalize.run, "desc": "S2 normalization of all 6 source files (parquet)"},
    "blocking": {"run": s3_blocking.run, "desc": "S3 multi-strategy blocking -> candidate pairs (train+test)"},
}
