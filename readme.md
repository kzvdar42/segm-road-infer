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
classes: cityscapes # which classes model has. Either mapillary or cityscapes
config_file: configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_tiny_bs16_90k.yaml # path to the detectron2 model config [for torch model]
opts: [] # You can ommit this)
weights_path: weights/sem-swin-t-cityscapes.pkl # path to the model weights
providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'] # ONNXRuntime providers
img_mean: null # Image mean used during model training
img_std: null # Image std used during model training. Set to [255, 255, 255] if need to normalize input
num_workers: 2 # Number of workers to create by DataLoader
input_shapes: # mapping between input shape and batch_size, if input shape outside this mapping is provided, it will use the batch size from the first key
  2048,1024: 5
  1024,512: 15
```

You can try to improve onnx runtime speed, by adding the `TensorrtExecutionProvider`, more info [here](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html).


## Run the script
To start inference, simply run the `run.sh` script like this:
```bash
./run.sh in_path out_path
```

By default it will use two models, one for ego vehicle extraction and the other one for everything else.
But you can also change the behaviour of the script by changing next environment variables:
* `MAIN_MODEL_CONFIG` - main model config filepath
* `MAIN_INPUT_SHAPE` - input shape for the main model
* `MAIN_BATCH_SIZE` - batch size for the main model
* `EGO_INPUT_SHAPE` - input shape for the ego model
* `EGO_BATCH_SIZE` - batch size for the ego model
* `EGO_MODEL_CONFIG` - ego model config filepath
* `SKIP_EGO_VEHICLE` - set to `1`, if you want to skip running ego vehicle model. Useful when running mapillary model, which already includes the ego vehicle class
* `EGO_MASK_PATH` - path to the folder to which script will save the results

Also, you can manually use the `infer.py` script, here is it's help command:
```bash
usage: python infer.py [-h] [--batch_size BATCH_SIZE] [--input_shape INPUT_SHAPE INPUT_SHAPE] [--out_format {mp4,jpg,png}] [--show] [--apply_ego_mask_from APPLY_EGO_MASK_FROM] [--n_skip_frames N_SKIP_FRAMES] [--only_ego_vehicle] [--skip_processed] [--test]
                       [--save_vis_to SAVE_VIS_TO] [--window_size WINDOW_SIZE WINDOW_SIZE] [--ffmpeg_setting_file FFMPEG_SETTING_FILE] [--no_tqdm]
                       model_config in_path out_path

positional arguments:
  model_config          path to the model yaml config
  in_path               path to input. It can be either folder/txt file with image paths/videofile.
  out_path              path to save the resulting masks

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        option to override model batch_size
  --input_shape INPUT_SHAPE INPUT_SHAPE
                        option to override model input shape
  --out_format {mp4,jpg,png}
                        format for saving the result
  --show                set to visualize predictions
  --apply_ego_mask_from APPLY_EGO_MASK_FROM
                        path to ego masks, will load them and apply to predictions
  --n_skip_frames N_SKIP_FRAMES
                        how many frames to skip during inference [default: 0]
  --only_ego_vehicle    store only ego vehicle class
  --skip_processed      skip already processed frames
  --test                test speed on 60 seconds runtime
  --save_vis_to SAVE_VIS_TO
                        path to save the visualized predictions. [default: None]
  --window_size WINDOW_SIZE WINDOW_SIZE
                        window size for visualization [default: (1280, 720)]
  --ffmpeg_setting_file FFMPEG_SETTING_FILE
                        path to ffmpeg settings. [default: `.ffmpeg_settings.yaml`]
  --no_tqdm             flag to not use tqdm progress bar
```

> :warning: Inferencing on images is currently slower than on video (approximately 30% slower)

## Infer results
If you ran model on images, the results will copy filenames and structure from input folder, but if you ran the script on videofile, results will be in `%05d.png` format.

Resulting masks will follow class ids from dataset on which model was trained ([class names](#other-info)) starting from 0. If you used `apply_ego_mask_from` flag running the script, masks will also have additional class for `ego vehicle` (it will be the last one).

## Speed Comparison
Ran test inference on WSL2 CPU: i5-9500 CPU @ 3.00GHz GPU: 2080ti

Used fp16 (for torch models), input shape 2048 1024 and increased batch_size.

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
* Classes list for corresponding datasets can be found in `datasets_meta` folder.

* Only models trained on `mapillary` dataset can be used for ego vehicle mask extraction.
