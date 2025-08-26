SESSION=$1
CONFIG=$2
TRAIN_FILE=$3

# Use default train file if not provided for backwards compatibility
DEFAULT_TRAIN_FILE="train.py"
TRAIN_FILE=${3:-$DEFAULT_TRAIN_FILE}

echo "Running with session: $SESSION, config: $CONFIG, train file: $TRAIN_FILE"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python scripts/$SESSION/$TRAIN_FILE $CONFIG