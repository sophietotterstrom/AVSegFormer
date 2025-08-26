SESSION="ms3" #$1
CONFIG="config/sav/AVSegFormer_pvt2_ms3_sav.py" #$2
#TRAIN_FILE="train.py"
TRAIN_FILE="sav_train.py"


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python scripts/$SESSION/$TRAIN_FILE $CONFIG
