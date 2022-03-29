# Segmentation Models Infer Script

Contributors:
* Vlad Kuleikin

## Installation
Follow `Mask2Former` [installation guide](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md)
After installation of `Mask2Former` set `MASK2FORMER_HOME` enviroment variable for the script to be able to load the Mask2Former model.

Also, you need to install `onnxruntime`, for this follow instructions from [here](https://onnxruntime.ai/docs/install/). You need to install the version with cuda support.

You can install `PyTurboJPEG` for faster jpeg loading, for this follow instructions from [here](https://github.com/lilohuang/PyTurboJPEG).

Run `pip install -r requirements.txt` to ensure that all needed packages are installed.

Update model configs with corresponding weights (for all models) and config files (for `mask2former` models).

<!-- To load onnx weights from repo, install [git-lfs](https://git-lfs.github.com/) and run `git lfs pull` -->

## Configure script
In the `configs` folder you can find config files for next models:
* `mask2former-r50-mapillary.yaml` - smaller mask2former *mapillary* model [\[weights\]](https://dl.fbaipublicfiles.com/maskformer/mask2former/mapillary_vistas/semantic/maskformer_R50_bs16_300k/model_final_6c66d0.pkl)
* `mask2former-swin-l-mapillary.yaml` - biggest mask2former *mapillary* model [\[weights\]](https://dl.fbaipublicfiles.com/maskformer/mask2former/mapillary_vistas/semantic/maskformer2_swin_large_IN21k_384_bs16_300k/model_final_90ee2d.pkl)
* `mask2former-r50-cityscapes.yaml` - smaller mask2former *cityscapes* model [\[weights\]](https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_R50_bs16_90k/model_final_cc1b1f.pkl)
* `mask2former-swin-t-cityscapes.yaml` - bigger mask2former *cityscapes* model [\[weights\]](https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_swin_tiny_bs16_90k/model_final_2d58d4.pkl)
* `mask2former-swin-b-cityscapes.yaml` - biggest mask2former *cityscapes* model [\[weights\]](https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_swin_base_IN21k_384_bs16_90k/model_final_1c6b65.pkl)
* `onnx-deeplabv3plus-r18d-dynamic.yaml` - slow onnx deeplabv3plus-r18d *cityscapes* model [\[weights\]](https://disk.yandex.ru/d/Mosp_kwsLMGZyQ)
* `onnx-bisectv1-dynamic.yaml` - fast onnx bisectv1 *cityscapes* model [\[weights\]](https://disk.yandex.ru/d/Ucsulu2D7_b27A)

Detailed info about configs:
```yaml
model_type: torch # model type. Either torch or onnx
image_format: rgb # input image format. Either rgb or bgr
use_fp16: True # Set true to use fp16 for inference. Increases speed, and possibly subtle decrease in quality [Only for torch model!]
classes: mapillary #  on which dataset model was trained. Either mapillary or cityscapes
config_file: Mask2Former/configs/mapillary-vistas/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml # path to the detectron2 model config [for torch model]
opts: [] # You can ommit this)
weights_path: weights/sem_swin_large_patch4_window12_384_22k.pkl # path to the model weights
providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'] # ONNXRuntime providers
input_shape: [2048, 1024] # model input shape: width, height
img_mean: null # Image mean used during model training
img_std: null # Image std used during model training. Set to [255, 255, 255] if need to normalize input
batch_size: 3 # Batch size to use by DataLoader
num_workers: 4 # Number of workers to create by DataLoader
```

You can try to improve onnx runtime speed, by adding the `TensorrtExecutionProvider`, more info [here](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html).


## Run the script
You can run the script using two models, one for ego vehicle mask, other for whole image segmentation. To do this, change variables in `run.sh`. After that you can simply run in like this:
```bash
./run.sh in_path out_path
```


Also, you can manually use the `infer.py` script, here is it's help command:
```bash
usage: python infer.py [-h] [--show] [--apply_ego_mask_from APPLY_EGO_MASK_FROM] [--n_skip_frames N_SKIP_FRAMES] [--only_ego_vehicle]
                       [--save_vis_to SAVE_VIS_TO] [--window_size WINDOW_SIZE [WINDOW_SIZE ...]]
                       model_config in_path out_path

positional arguments:
  model_config          path to the model yaml config
  in_path               path to input folder with images/input videofile. Will either read all images under this path, or load provided
                        videofile
  out_path              path to save the resulting masks

optional arguments:
  -h, --help            show this help message and exit
  --show                set to visualize predictions
  --apply_ego_mask_from APPLY_EGO_MASK_FROM
                        path to ego masks, will load them and apply to predictions
  --n_skip_frames N_SKIP_FRAMES
                        how many frames to skip during inference [default: 0]
  --only_ego_vehicle    store only ego vehicle class
  --save_vis_to SAVE_VIS_TO
                        path to save the visualized predictions. [default: None]
  --window_size WINDOW_SIZE [WINDOW_SIZE ...]
                        window size for visualization
```

> :warning: Inferencing on videofile is currently slow (approximately 2x times slower)

## Infer results
If you ran model on images, the results will copy filenames and structure from input folder, but if you ran the script on videofile, results will be in `%05d.png` format.

Resulting masks will follow class ids from dataset on which model was trained ([class names](#other-info)) starting from 0. If you used `apply_ego_mask_from` flag running the script, masks will also have additional class for `ego vehicle` (it will be the last one).

## Speed Comparison
Ran test inference on WSL2 CPU: i5-9500 CPU @ 3.00GHz GPU: 2080ti

Used fp16 (for torch models) and increased batch_size

### Mapillary dataset *(sorted by speed)*
| Model              | fps   | mIoU	| mIoU (ms+flip) | model type |
|:------------------ |:-----:|:----:|:--------------:|:----------:|
| mask2former-r50    | 5.05  | 57.4 | 59.0           | torch      |
| mask2former-swin-l | 2.54  | 63.2 | 64.7           | torch      |

### Cityscapes dataset *(sorted by speed)*

| Model                 | fps   | mIoU	| mIoU (ms+flip) | model type |
|:--------------------- |:-----:|:-----:|:--------------:|:----------:|
| BiSeNetV1 	          | 18	  | 75.16 | 77.24          | onnx       |
| BiSeNetV2 (FP16)      | 18    | 73.07 | 75.13          | onnx       |
| FastSCNN              | 10    | 70.96 | 72.65          | onnx       |
| DeepLabV3+	R-18-D8	  | 10.98 | 76.89 | 78.76          | onnx       |
| mask2former-r50       | 5.17  | 79.4  | 82.2           | torch      |
| mask2former-swin-t    | 4.46  | 82.1  | 83.0           | torch      |
| mask2former-swin-b    | 3.33  | 83.3  | 84.5           | torch      |
| DeepLabV3+	R-101b-D8 | 2.60	| 80.16 | 81.41          | onnx       |

## Other info
* Classes list for corresponding datasets can be found in `cityscapes-classes.txt` and `mapillary-classes.txt`.

* Only models trained on `mapillary` dataset can be used for ego vehicle mask extraction.
