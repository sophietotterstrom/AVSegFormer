SESSION="ms3"
TRAIN_FILE="sav_train.py"

# Use default config if not provided
DEFAULT_CONFIG="config/sav/AVSegFormer_pvt2_ms3_sav.py"
CONFIG=${1:-$DEFAULT_CONFIG}

echo "Running with session: $SESSION, config: $CONFIG"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python scripts/$SESSION/$TRAIN_FILE $CONFIG