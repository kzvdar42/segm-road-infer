# Define enviroment variables
export MASK2FORMER_HOME=/mnt/c/Users/Vlad/Desktop/ny-guiderails/Mask2Former
export PYTHONWARNINGS="ignore"
EGO_MASK_PATH="ego-vehicle-mask/"
# Ensure that script hast two inputs: in_path, out_path. Otherwise exit
if [ "$#" -ne 2 ]; then
  echo "Need two arguments: in_path out_path"
  exit 1
fi
# Calculate ego vehicle masks
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~ Calculating ego vehicle masks ~~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
rm -rf ego-vehicle-mask
python infer.py configs/mask2former-swin-l-mapillary.yaml $1 $EGO_MASK_PATH --only_ego_vehicle --n_skip_frames 100
# Calculate segmentation masks
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~ Calculating segmentation masks ~~~~~~~~~~~~~"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
python infer.py configs/onnx-bisectv1-dynamic.yaml $1 $2 --apply_ego_mask_from $EGO_MASK_PATH

# Other onnx models
# python infer.py configs/onnx-deeplabv3plus-r18d-dynamic.yaml $1 $2 --apply_ego_mask_from ego-vehicle-mask
# python infer.py configs/onnx-bisectv2-fp16-dynamic.yaml $1 $2 --apply_ego_mask_from ego-vehicle-mask

# Slower, torch models
# python infer.py configs/mask2former-r50-cityscapes.yaml $1 $2
# python infer.py configs/mask2former-swin-l-mapillary.yaml $1 $2