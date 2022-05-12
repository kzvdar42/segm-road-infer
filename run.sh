# set -e for immediate exit if one of the commands exits with non-zero code
set -e
# Define enviroment variables
if [ -z "$MASK2FORMER_HOME" ]; then
  echo "Warning: MASK2FORMER_HOME env var is not set and will be set to a default value"
  export MASK2FORMER_HOME=/mnt/c/Users/Vlad/Desktop/ny-guiderails/Mask2Former
fi
export PYTHONWARNINGS="ignore"
EGO_MASK_PATH="ego-vehicle-mask/"
# Ensure that script hast two inputs: in_path, out_path. Otherwise exit
if [ "$#" -ne 2 ]; then
  echo "Need two arguments: in_path out_path"
  exit 1
fi

INPUT_SHAPE_ARG=""
if [ ! -z "$INPUT_SHAPE" ]; then
  INPUT_SHAPE_ARG="--input_shape $INPUT_SHAPE"
fi

MAIN_BATCH_SIZE_ARG=""
if [ ! -z "$MAIN_BATCH_SIZE" ]; then
  MAIN_BATCH_SIZE_ARG="--batch_size $MAIN_BATCH_SIZE"
fi

EGO_BATCH_SIZE_ARG=""
if [ ! -z "$EGO_BATCH_SIZE" ]; then
  EGO_BATCH_SIZE_ARG="--batch_size $EGO_BATCH_SIZE"
fi

if [ -z "$MAIN_MODEL_CONFIG" ]; then
  MAIN_MODEL_CONFIG="configs/onnx-bisectv1-dynamic.yaml"
fi

if [ -z "$EGO_MODEL_CONFIG" ]; then
  EGO_MODEL_CONFIG="configs/mask2former-r50-mapillary.yaml"
fi

set -x

# Calculate ego vehicle masks
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~ Calculating ego vehicle masks ~~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
rm -rf $EGO_MASK_PATH
mkdir $EGO_MASK_PATH
python infer.py $EGO_MODEL_CONFIG $1 $EGO_MASK_PATH --only_ego_vehicle --n_skip_frames -2 --out_format png --no_tqdm $INPUT_SHAPE_ARG $EGO_BATCH_SIZE_ARG #  --n_skip_frames -2 - take one frame per 2 seconds
# Calculate segmentation masks
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~ Calculating segmentation masks ~~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
python infer.py $MAIN_MODEL_CONFIG $1 $2 --apply_ego_mask_from $EGO_MASK_PATH --no_tqdm $INPUT_SHAPE_ARG $MAIN_BATCH_SIZE_ARG # --input_shape 1024 512

# Other onnx models
# python infer.py configs/onnx-deeplabv3plus-r18d-dynamic.yaml $1 $2 --apply_ego_mask_from $EGO_MASK_PATH
# python infer.py configs/onnx-bisectv2-fp16-dynamic.yaml $1 $2 --apply_ego_mask_from $EGO_MASK_PATH

# Slower, torch models
# python infer.py configs/mask2former-r50-cityscapes.yaml $1 $2 --apply_ego_mask_from $EGO_MASK_PATH
# python infer.py configs/mask2former-swin-b-cityscapes.yaml $1 $2 --apply_ego_mask_from $EGO_MASK_PATH