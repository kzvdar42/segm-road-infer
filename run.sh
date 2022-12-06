# set -e for immediate exit if one of the commands exits with non-zero code
set -e
# Define enviroment variables
if [ -z "$MASK2FORMER_HOME" ]; then
  echo "Warning: MASK2FORMER_HOME env var is not set and will be set to a default value"
  export MASK2FORMER_HOME=/mnt/c/Users/Vlad/Desktop/hdmaps-dev/slam_segmentation_inference/Mask2Former
fi
export PYTHONWARNINGS="ignore"
export ORT_TENSORRT_DLA_ENABLE=1
export ORT_TENSORRT_ENGINE_CACHE_ENABLE=1
export ORT_TENSORRT_CACHE_PATH="tensorrt_cache/"
# export ORT_TENSORRT_FP16_ENABLE=1
EGO_MASK_PATH="ego-vehicle-mask/"
# Ensure that script hast two inputs: in_path, out_path. Otherwise exit
if [ "$#" -ne 2 ]; then
  echo "Need two arguments: in_path out_path"
  exit 1
fi

MAIN_INPUT_SHAPE_ARG=""
if [ ! -z "$MAIN_INPUT_SHAPE" ]; then
  MAIN_INPUT_SHAPE_ARG="--input_shape $MAIN_INPUT_SHAPE"
fi

EGO_INPUT_SHAPE_ARG=""
if [ ! -z "$EGO_INPUT_SHAPE" ]; then
  EGO_INPUT_SHAPE_ARG="--input_shape $EGO_INPUT_SHAPE"
fi

MAIN_BATCH_SIZE_ARG=""
if [ ! -z "$MAIN_BATCH_SIZE" ]; then
  MAIN_BATCH_SIZE_ARG="--batch_size $MAIN_BATCH_SIZE"
fi

EGO_BATCH_SIZE_ARG=""
if [ ! -z "$EGO_BATCH_SIZE" ]; then
  EGO_BATCH_SIZE_ARG="--batch_size $EGO_BATCH_SIZE"
fi

CLS_MAPPING_OUT_ARG=""
if [ ! -z "$CLS_MAPPING_OUT_PATH" ]; then
  CLS_MAPPING_OUT_ARG="--out_cls_mapping $CLS_MAPPING_OUT_PATH"
fi

if [ -z "$MAIN_MODEL_CONFIG" ]; then
  MAIN_MODEL_CONFIG="configs/onnx-bisectv1-dynamic.yaml"
fi

if [ -z "$EGO_MODEL_CONFIG" ]; then
  EGO_MODEL_CONFIG="configs/mask2former-r50-mapillary.yaml"
fi

set -x

# Calculate ego vehicle masks
APPLY_EGO_MASK_FROM=""
if [ -z "$SKIP_EGO_VEHICLE" ] || [ "$SKIP_EGO_VEHICLE" = 0 ]; then
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "~~~~~~~~~~~~~ Calculating ego vehicle masks ~~~~~~~~~~~~~"
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  rm -rf $EGO_MASK_PATH
  mkdir $EGO_MASK_PATH
  python infer.py $EGO_MODEL_CONFIG $1 $EGO_MASK_PATH --only_ego_vehicle \
         --n_skip_frames -2 --out_format png --no_tqdm $EGO_INPUT_SHAPE_ARG \
         $EGO_BATCH_SIZE_ARG
  #  --n_skip_frames -2 - take one frame per 2 seconds
  APPLY_EGO_MASK_FROM="--apply_ego_mask_from $EGO_MASK_PATH"
fi
# Calculate segmentation masks
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~ Calculating segmentation masks ~~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
python infer.py $MAIN_MODEL_CONFIG $1 $2 $APPLY_EGO_MASK_FROM --no_tqdm \
  $MAIN_INPUT_SHAPE_ARG $MAIN_BATCH_SIZE_ARG $CLS_MAPPING_OUT_ARG

# Other onnx models
# python infer.py configs/onnx-deeplabv3plus-r18d-dynamic.yaml $1 $2 --apply_ego_mask_from $EGO_MASK_PATH
# python infer.py configs/onnx-bisectv2-fp16-dynamic.yaml $1 $2 --apply_ego_mask_from $EGO_MASK_PATH

# Slower, torch models
# python infer.py configs/mask2former-r50-cityscapes.yaml $1 $2 --apply_ego_mask_from $EGO_MASK_PATH
# python infer.py configs/mask2former-swin-b-cityscapes.yaml $1 $2 --apply_ego_mask_from $EGO_MASK_PATH