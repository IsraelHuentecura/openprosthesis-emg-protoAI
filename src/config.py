from pathlib import Path

# --- Project Paths ---
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RUNS_DIR = ROOT_DIR / "runs"
MODELS_DIR = ROOT_DIR / "models"
BACKUP_DIR = RUNS_DIR / "_training_backups"

# --- Dataset & Preprocessing Parameters ---
# These should match the parameters used in your preprocessing script.
FS = 100  # Sampling frequency
WIN_LEN = 20  # Window length in samples (200 ms)
T_SUBWIN = 5  # Number of windows per sequence
HANDCRAFT_PER_CH = 10  # Number of handcrafted features per channel
N_CHANNELS = 10 # Number of sEMG channels in Ninapro DB1
RAW_SHAPE = (WIN_LEN, N_CHANNELS, 1)
FEAT_DIM = N_CHANNELS * HANDCRAFT_PER_CH
NUM_CLASSES = 12 # 12 gestures from Stimulus 1

# --- Training Parameters ---
SEED = 42
N_FOLDS = 10
EPOCHS = 100
BATCH_SIZE = 128
TEST_SPLIT_PCT = 0.20

# --- Model Hyperparameters (Defaults) ---
# These can be overridden when calling the build functions.
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_DROPOUT_RATE = 0.3