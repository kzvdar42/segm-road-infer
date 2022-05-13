import argparse
import os
import subprocess
import time
from typing import List, Tuple
import yaml

import addict
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils import (
    PseudoTqdm, is_image, load_model_config,
    colorize_pred, apply_mask, create_folder_for_file, get_classes,
    get_out_path
)
from infer_utils import (
    FfmpegVideoDataset, Predictor, ONNXPredictor, DetectronPredictor, ImageDataset, VideoDataset, ffmpeg_start_out_imgs_process,
    ffmpeg_start_out_process, index_images
)


def load_predictor(model_cfg: dict) -> Predictor:
    model_type = model_cfg.pop('model_type')
    if model_type == 'onnx':
        predictor = ONNXPredictor(model_cfg)
    elif model_type == 'torch':
        predictor = DetectronPredictor(model_cfg)
    else:
        raise ValueError(f'Unknown model type ({model_type})')
    return predictor


def show_preds(predictions: List[np.ndarray], metadata: dict, show: bool = True,
               save_to: str = None, window_size: Tuple[int, int] = None) -> bool:
    window_size = window_size or (1280, 720)
    for pred, meta in zip(predictions, metadata):
        img_name = os.path.split(meta['image_path'])[1]
        img_orig = cv2.imread(meta['image_path'])
        res_img = colorize_pred(pred)
        res_img = apply_mask(img_orig, res_img, alpha=0.5)
        if save_to is not None:
            out_path = os.path.join(save_to, img_name)
            create_folder_for_file(out_path)
            cv2.imwrite(out_path, res_img)
            return False
        if show:
            cv2.imshow('pred_vis', cv2.resize(res_img, window_size))
            # esc to quit
            return cv2.waitKey(0) == 27
        return False


def save_preds_as_masks(predictions: List[np.ndarray], metadata: dict, in_base_path: str, out_path: str, ext: str) -> None:
    for pred, meta in zip(predictions, metadata):
        pred_out_path = get_out_path(meta['image_path'], out_path, in_base_path, ext)
        create_folder_for_file(pred_out_path)
        if not cv2.imwrite(pred_out_path, pred):
            print(f'Didn\'t save!', pred_out_path, type(pred), pred.shape)


def save_preds_to_ffmpeg(predictions: List[np.ndarray], video_writer: subprocess.Popen) -> None:
    video_writer.stdin.write(predictions.tobytes())


def default_ffmpeg_args():
    return dict(
        out_vcodec = "libx264",
        out_pix_fmt = "yuv420p",
        output_args = dict(crf=0),
        in_global_args = "-hide_banner -loglevel error",
        out_global_args = "-hide_banner -loglevel error",
        max_timeout = 15,
    )


def get_args():
    parser = argparse.ArgumentParser('python infer.py')
    parser.add_argument('model_config', help='path to the model yaml config')
    parser.add_argument('in_path', help='path to input. It can be either folder/txt file with image paths/videofile.')
    parser.add_argument('out_path', help='path to save the resulting masks')
    parser.add_argument('--batch_size', type=int, default=None, help='option to override model batch_size')
    parser.add_argument('--input_shape', type=int, nargs=2, default=None, help='option to override model input shape')
    parser.add_argument('--out_format', default='mp4', choices=['mp4', 'jpg', 'png'], help='format for saving the result')
    parser.add_argument('--show', action='store_true', help='set to visualize predictions')
    parser.add_argument('--apply_ego_mask_from', help='path to ego masks, will load them and apply to predictions')
    parser.add_argument('--n_skip_frames', type=int, default=0, help='how many frames to skip during inference [default: 0]')
    parser.add_argument('--only_ego_vehicle', action='store_true', help='store only ego vehicle class')
    parser.add_argument('--skip_processed', action='store_true', help='skip already processed frames')
    parser.add_argument('--test', action='store_true', help='test speed on 60 seconds runtime')
    parser.add_argument('--save_vis_to', default=None, help='path to save the visualized predictions. [default: None]')
    parser.add_argument('--window_size', type=int, nargs=2, default=(1280, 720), help='window size for visualization [default: (1280, 720)]')
    parser.add_argument('--ffmpeg_setting_file', default='.ffmpeg_settings.yaml', help='path to ffmpeg settings. [default: `.ffmpeg_settings.yaml`]')
    parser.add_argument('--no_tqdm', action='store_true', help='flag to not use tqdm progress bar')

    args = parser.parse_args()
    args.print_every_n = 500

    if args.out_path.endswith('.mp4'):
        args.out_format = 'mp4'
    elif os.path.isdir(args.out_path) and args.out_format == 'mp4':
        raise argparse.ArgumentError('Either provide out_path with video name or choose another out_format!')
    
    # if n_skip_frames < 0, set automatically to 1 fps
    if args.n_skip_frames < 0:
        if not args.in_path.endswith('.mp4'):
            args.n_skip_frames = 30
        args.n_skip_frames = abs(args.n_skip_frames) * FfmpegVideoDataset.get_video_metadata(args.in_path)[-2]

    args.ffmpeg = default_ffmpeg_args()
    if os.path.isfile(args.ffmpeg_setting_file):
        print(f'Loading ffmpeg settings from {args.ffmpeg_setting_file}')
        with open(args.ffmpeg_setting_file) as in_stream:
            args.ffmpeg = yaml.safe_load(in_stream)
    
    # load model config
    args.model_cfg = load_model_config(args.model_config)
    if args.input_shape:
        args.model_cfg.input_shape = args.input_shape
        if args.batch_size:
            args.model_cfg.batch_size = args.batch_size
        else:
            default_batch_size = list(args.model_cfg.input_shapes.values())[0]
            args.model_cfg.batch_size = args.model_cfg.input_shapes.get(
                ','.join(map(str,args.input_shape)), default_batch_size
            )
    else:
        # Take first value as default
        input_shape = next(iter(args.model_cfg.input_shapes.keys()))
        args.model_cfg.batch_size = args.model_cfg.input_shapes[input_shape]
        args.model_cfg.input_shape = tuple(int(side) for side in input_shape.split(','))
    # make print divisible by batch_size
    args.print_every_n = (args.print_every_n // args.model_cfg.batch_size) * args.model_cfg.batch_size

    args = addict.Dict(vars(args))
    return args


if __name__ == '__main__':
    # Get args & load model config
    script_start_time = time.time()
    torch.backends.cudnn.benchmark = True
    args = get_args()
    print('Args:', args, sep='\n')

    # Load classes
    classes, cls_name_to_id, cls_id_to_name = get_classes(args.model_cfg.classes)
    if args.only_ego_vehicle:
        ego_class_ids = np.array([cls_name_to_id[cls_name] for cls_name in ['ego vehicle', 'car mount']])
        ego_class_ids = torch.from_numpy(ego_class_ids).to(device='cuda', dtype=torch.uint8)[None, ...]
        assert ego_class_ids.shape[-1], 'Model without ego vehicle classes!'

    # Create dataloader (if folder/image/txt_file -> image dataset, otherwise video)
    if os.path.isdir(args.in_path) or is_image(args.in_path) or args.in_path.endswith('.txt'):
        image_paths = index_images(args)
        dataset = ImageDataset(image_paths, args)
    else:
        args.base_path = os.path.abspath(os.path.split(args.in_path)[0])
        if args.skip_processed:
            print('[WARNING] skip_processed flag is not yet supported for video inputs!')
        # dataset = VideoDataset(args.in_path, args, args.n_skip_frames)
        if args.n_skip_frames:
            dataset = VideoDataset(args.in_path, args, args.n_skip_frames)
        else:
            dataset = FfmpegVideoDataset(args.in_path, args, args.n_skip_frames)
            args.model_cfg.num_workers = 1
        print(f'Loaded video, total frames {dataset.len}')
        if args.n_skip_frames:
            print(f'Skipping {args.n_skip_frames} frames each time, {len(dataset)} left')

    dataloader = DataLoader(
        dataset, batch_size=args.model_cfg.batch_size, pin_memory=True,
        num_workers=args.model_cfg.num_workers, collate_fn=dataset.collate_fn,
        shuffle=False, prefetch_factor=8,
    )

    # Create videowriter if saving as a video
    if args.out_format == 'mp4':
        out_width, out_height, fps = 1920, 1080, 30
        if isinstance(dataset, (FfmpegVideoDataset, VideoDataset)):
            out_width, out_height, fps =  dataset.orig_width, dataset.orig_height, dataset.fps
        out_writer = ffmpeg_start_out_process(
            args.ffmpeg, args.out_path, *args.model_cfg.input_shape, out_width, out_height, fps
        )
        print(f'Saving results to videofile with {fps} fps and {(out_width, out_height)} frame size.')
    # else:
    #     out_writer = ffmpeg_start_out_imgs_process(
    #         args.ffmpeg, args.out_path, args.out_format, *args.model_cfg.input_shape
    #     )

    # Load model
    model = load_predictor(args.model_cfg)
    print(f'Infer image size: {args.model_cfg.input_shape}; batch size: {args.model_cfg.batch_size}')

    # Test pre-run
    if args.test:
        model(torch.rand((args.model_cfg.batch_size, 3, *args.model_cfg.input_shape), device='cuda'))

    # Infer model
    # XXX: Using pbar like this, because otherwise ffmpeg subprocess wouldn't finish
    pbar = tqdm(total=len(dataset)) if not args.no_tqdm else PseudoTqdm()
    resize_img = args.out_format != 'mp4' and not args.only_ego_vehicle
    for images, img_masks, metadata in dataloader:
        if args.test and time.time() - pbar.start_t > 60:
            break
        images = images.to('cuda', non_blocking=True)
        if img_masks is not None:
            img_masks = img_masks.to('cuda', non_blocking=True)
        predictions = model(images)
        predictions = dataset.postprocess(
            predictions, metadata, img_masks=img_masks, ego_mask_cls_id=len(classes), resize_img=resize_img
        )

        if args.only_ego_vehicle:
            predictions = ((predictions[..., None].byte() == ego_class_ids).any(-1) * 255)
            # predictions = torch.isin(predictions.byte(), ego_class_ids[0]) * 255

        # Transfer to numpy
        predictions = predictions.byte().cpu().numpy()

        if args.show or args.save_vis_to:
            # If user exits, destroy all windows and break
            if show_preds(predictions, metadata, args.show, args.save_vis_to, args.window_size):
                cv2.destroyAllWindows()
                break

        if args.out_path:
            if args.out_format in ['png', 'jpg']:
                # save_preds_to_ffmpeg(predictions, out_writer)
                save_preds_as_masks(predictions, metadata, args.base_path, args.out_path, args.out_format)
            elif args.out_format == 'mp4':
                save_preds_to_ffmpeg(predictions, out_writer)
            else:
                raise ValueError(f'Unknown out format! ({args.out_format})')
        
        pbar.update(images.shape[0])
        if args.no_tqdm:
            if pbar.n_runs % args.print_every_n == 0:
                print(f'Processed {pbar.n_runs} images, at rate {pbar.rate:.2f} imgs/s', flush=True)

    if args.no_tqdm:
        print(f'Processed {pbar.n_runs} images, at rate {pbar.rate:.2f} imgs/s. Total: {time.time() - pbar.start_t:.2f} sec')
    pbar.close()

    # Exit from or kill ffmpeg processes
    if args.out_format == 'mp4':
        print('Waiting for ffmpeg to exit...')
        try:
            out_writer.communicate(timeout=args.ffmpeg.max_timeout)
        except subprocess.TimeoutExpired:
            print(f'Waited for {args.ffmpeg.max_timeout} seconds. Killing ffmpeg!')
            out_writer.kill()
            out_writer.communicate()
            print(f'ffmpeg killed with code {out_writer.returncode}')
        else:
            out_writer.communicate()
            print(f'ffmpeg succesfully exited with code {out_writer.returncode}')
    print(f'Total script time: {time.time() - script_start_time:.2f}')
