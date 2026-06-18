from pathlib import Path

# Root of the ARGOS repository
ROOT_DIR = Path(__file__).resolve().parent.parent

# Core subdirectories
SCRIPTS_DIR = ROOT_DIR / "scripts"
EXTERNAL_DIR = ROOT_DIR / "external"
DATASET_DIR = ROOT_DIR / "dataset"
RESULTS_DIR = ROOT_DIR / "results"
CONFIGS_DIR = ROOT_DIR / "configs"

# Common dataset locations
SCARED_DIR = DATASET_DIR / "SCARED"
SERVCT_DIR = DATASET_DIR / "SERVCT"
D4D_DIR = DATASET_DIR / "D4D"

# Python environments
ARGOS_ENV_PYTHON = ROOT_DIR / ".miniconda" / "envs" / "argos" / "bin" / "python"

# Helper function to get paths relative to root
def get_path(relative_path: str) -> Path:
    return ROOT_DIR / relative_path
