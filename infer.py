import argparse
import os
from socket import timeout
import subprocess
from typing import List, Tuple
import yaml

import addict
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils import (
    is_image, load_model_config,
    colorize_pred, apply_mask, create_folder_for_file, get_classes,
    get_out_path
)
from infer_utils import (
    FfmpegVideoDataset, Predictor, ONNXPredictor, DetectronPredictor, ImageDataset, VideoDataset,
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


def save_preds_as_video(predictions: List[np.ndarray], video_writer: cv2.VideoWriter) -> None:
    for pred in predictions:
        video_writer.stdin.write(pred.tobytes())


def default_ffmpeg_args():
    return dict(
        vcodec = "libx264",
        pix_fmt = "yuv420p",
        output_args = dict(crf=0),
        global_args = "-hide_banner -loglevel error",
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
    parser.add_argument('--save_vis_to', default=None, help='path to save the visualized predictions. [default: None]')
    parser.add_argument('--window_size', type=int, nargs=2, default=(1280, 720), help='window size for visualization [default: (1280, 720)]')
    parser.add_argument('--ffmpeg_setting_file', default='.ffmpeg_settings.yaml', help='path to ffmpeg settings. [default: `.ffmpeg_settings.yaml`]')
    parser.add_argument('--no_tqdm', action='store_true', help='flag to not use tqdm progress bar')

    args = parser.parse_args()
    args.print_every_n = 5

    if args.out_path.endswith('.mp4'):
        args.out_format = 'mp4'
    elif os.path.isdir(args.out_path) and args.out_format == 'mp4':
        raise argparse.ArgumentError('Either provide out_path with video name or choose another out_format!')
    
    # if n_skip_frames < 0, set automatically to 1 fps
    if args.n_skip_frames < 0:
        if not args.in_path.endswith('.mp4'):
            args.n_skip_frames = 30
        args.n_skip_frames = FfmpegVideoDataset.get_video_metadata(args.in_path)[-1]

    args.ffmpeg = default_ffmpeg_args()
    if os.path.isfile(args.ffmpeg_setting_file):
        with open(args.ffmpeg_setting_file) as in_stream:
            args.ffmpeg = yaml.safe_load(in_stream)

    args = addict.Dict(vars(args))
    return args


if __name__ == '__main__':
    # Get args & load model config
    args = get_args()
    args.model_cfg = load_model_config(args.model_config)
    if args.batch_size:
        args.model_cfg.batch_size = args.batch_size
    if args.input_shape:
        args.model_cfg.input_shape = args.input_shape
    print('Args:', args, sep='\n')

    # Load classes
    classes, cls_name_to_id, cls_id_to_name = get_classes(args.model_cfg.classes)
    if args.only_ego_vehicle:
        ego_class_ids = [cls_name_to_id[cls_name] for cls_name in ['ego vehicle', 'car mount']]
        assert len(ego_class_ids), 'Model without ego vehicle classes!'

    # Create dataloader (if folder/image/txt_file -> image dataset, otherwise video)
    if os.path.isdir(args.in_path) or is_image(args.in_path) or args.in_path.endswith('.txt'):
        image_paths = index_images(args)
        dataset = ImageDataset(image_paths, args)
    else:
        args.base_path = os.path.abspath(os.path.split(args.in_path)[0])
        if args.skip_processed:
            print('[WARNING] skip_processed flag is not yet supported for video inputs!')
        if args.n_skip_frames:
            dataset = VideoDataset(args.in_path, args, args.n_skip_frames)
        else:
            dataset = FfmpegVideoDataset(args.in_path, args, args.n_skip_frames)
            args.model_cfg.num_workers = 1
        print(f'Loaded video, total frames {dataset.len}')
        if args.n_skip_frames:
            print(f'Skipping {args.n_skip_frames} frames each time, {len(dataset)} left')

    dataloader = DataLoader(
        dataset, batch_size=args.model_cfg.batch_size, pin_memory=args.model_cfg.model_type == 'torch',
        num_workers=args.model_cfg.num_workers, collate_fn=dataset.collate_fn,
        shuffle=False, prefetch_factor=8,
    )

    # Create videowriter if saving as a video
    if args.out_path and args.out_format == 'mp4':
        out_width, out_height, fps = 1920, 1080, 30
        if isinstance(dataset, (FfmpegVideoDataset, VideoDataset)):
            out_width, out_height, fps =  dataset.orig_width, dataset.orig_height, dataset.fps
        video_writer = ffmpeg_start_out_process(
            args.ffmpeg, args.out_path, *args.model_cfg.input_shape, out_width, out_height, fps
        )
        print(f'Saving results to videofile with {fps} fps and {(out_width, out_height)} frame size.')

    # Load model
    model = load_predictor(args.model_cfg)

    # Infer model
    # XXX: Using pbar like this, because otherwise ffmpeg subprocess wouldn't finish
    if not args.no_tqdm:
        pbar = tqdm(total=len(dataloader))
    for n_batch, (images, metadata) in enumerate(dataloader, 1):
        predictions = model(images)
        predictions = dataset.postprocess(
            predictions, metadata, ego_mask_cls_id=len(classes), resize_img=args.out_format != 'mp4'
        )

        if args.only_ego_vehicle:
            for pred_num, pred in enumerate(predictions):
                predictions[pred_num] = (np.isin(pred, ego_class_ids) * 255).reshape(pred.shape)

        if args.show or args.save_vis_to:
            # If user exits, destroy all windows and break
            if show_preds(predictions, metadata, args.show, args.save_vis_to, args.window_size):
                cv2.destroyAllWindows()
                break  

        if args.out_path:
            if args.out_format in ['png', 'jpg']:
                save_preds_as_masks(predictions, metadata, args.base_path, args.out_path, args.out_format)
            elif args.out_format == 'mp4':
                save_preds_as_video(predictions, video_writer)
            else:
                raise ValueError(f'Unknown out format! ({args.out_format})')
        if args.no_tqdm:
            if n_batch % args.print_every_n == 0:
                print(f'Processed {n_batch} batches')
        else:
            pbar.update(1)

    if not args.no_tqdm:
        pbar.close()

    # Exit from or kill ffmpeg processes
    if args.out_format == 'mp4':
        print('Waiting for ffmpeg to exit...')
        try:
            video_writer.communicate(timeout=args.ffmpeg.max_timeout)
        except subprocess.TimeoutExpired:
            print(f'Waited for {args.ffmpeg.max_timeout} seconds. Killing ffmpeg!')
            video_writer.kill()
            video_writer.communicate()
            print(f'ffmpeg killed with code {video_writer.returncode}')
        else:
            video_writer.communicate()
            print(f'ffmpeg succesfully exited with code {video_writer.returncode}')
