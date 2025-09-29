SESSION=$1
CONFIG=$2
WEIGHTS=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_LAUNCH_BLOCKING=1 python scripts/$SESSION/test.py \
        $CONFIG \
        $WEIGHTS \
        --save_pred_mask \
        --save_dir "/scratch/project_2005102/sophie/output"
