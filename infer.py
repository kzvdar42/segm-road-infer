import argparse
import os
import subprocess
import time
import yaml

import addict
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.infer import load_predictor
from src.datasets import FfmpegVideoDataset, load_dataset
from src.utils.infer import (
    PseudoTqdm, load_model_config, default_ffmpeg_args,
    create_out_writer, show_preds, save_preds_to_ffmpeg, save_preds_as_masks
)
from src.utils.datasets_meta import get_classes, get_palette


def infer_model(model, dataloader, args, out_writer):
    # XXX: Using pbar like this, because otherwise ffmpeg subprocess wouldn't finish
    pbar = tqdm(total=len(dataset)) if not args.no_tqdm else PseudoTqdm()
    resize_img = args.out_format != 'mp4' and not args.only_ego_vehicle
    stats = dict(data_load=[], model_infer=[], postprocess=[], save=[])
    last_time = time.time()
    for images, img_masks, metadata in dataloader:
        if args.test and time.time() - pbar.start_t > 60:
            break
        images = images.to('cuda', non_blocking=True)
        if img_masks is not None:
            img_masks = img_masks.to('cuda', non_blocking=True)
        stats['data_load'].append(time.time()-last_time)
        last_time = time.time()
        predictions = model(images)
        stats['model_infer'].append(time.time()-last_time)
        last_time = time.time()
        predictions = dataset.postprocess(
            predictions, metadata, img_masks=img_masks, ego_mask_cls_id=len(classes), resize_img=resize_img
        )

        if args.only_ego_vehicle:
            predictions = ((predictions[..., None].byte() == ego_class_ids).any(-1) * 255)

        # Transfer to numpy
        predictions = predictions.byte().cpu().numpy()
        stats['postprocess'].append(time.time()-last_time)
        last_time = time.time()

        if args.show_or_save_mask:
            # If user exits, destroy all windows and break
            if show_preds(predictions, metadata, args.cls_palette, args.show, args.save_vis_to, args.window_size):
                cv2.destroyAllWindows()
                break

        if args.out_path:
            if args.out_format == 'mp4':
                save_preds_to_ffmpeg(predictions, out_writer)
                # save_preds_to_ffmpeg_with_palette(predictions, metadata, args.cls_palette, out_writer)
            elif args.out_format in ['png', 'jpg']:
                save_preds_as_masks(predictions, metadata, args.base_path, args.out_path, args.cls_palette, args.out_format)
            else:
                raise ValueError(f'Unknown out format! ({args.out_format})')
        stats['save'].append(time.time()-last_time)
        last_time = time.time()
        
        pbar.update(images.shape[0])
        if args.no_tqdm:
            if pbar.n_runs % args.print_every_n == 0:
                print(f'Processed {pbar.n_runs} images, at rate {pbar.rate:.2f} imgs/s', flush=True)
    import numpy as np
    print({k:np.mean(v) for k, v in stats.items()})
    print({k:1/np.mean(v) for k, v in stats.items()})
    if args.no_tqdm:
        print(f'Processed {pbar.n_runs} images, at rate {pbar.rate:.2f} imgs/s. Total: {time.time() - pbar.start_t:.2f} sec')
    pbar.close()


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
    args.show_or_save_mask = args.show or args.save_vis_to

    if args.out_path.endswith('.mp4'):
        args.out_format = 'mp4'
    elif os.path.isdir(args.out_path) and args.out_format == 'mp4':
        raise argparse.ArgumentError('Either provide out_path with video name or choose another out_format!')
    
    # if n_skip_frames < 0, set automatically to 1 fps
    if args.n_skip_frames < 0:
        if not args.in_path.endswith('.mp4'):
            args.n_skip_frames = 30
        args.n_skip_frames = abs(args.n_skip_frames) * round(FfmpegVideoDataset.get_video_metadata(args.in_path)[-2])

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
    script_start_time = time.time()
    import multiprocessing as mp
    mp.set_start_method('fork')
    torch.backends.cudnn.benchmark = True

    # Get args & load model config
    args = get_args()
    print('Args:', args, sep='\n')

    # Load classes palette
    args.cls_palette = get_palette(
        'ego_data' if args.only_ego_vehicle else args.model_cfg.classes
    )

    # Load classes
    classes, cls_name_to_id, cls_id_to_name, ego_class_ids = get_classes(args.model_cfg.classes)
    if args.only_ego_vehicle:
        ego_class_ids = torch.from_numpy(ego_class_ids).to(device='cuda', dtype=torch.uint8)[None, ...]
        assert ego_class_ids.shape[-1], 'Model without ego vehicle classes!'

    # Load dataset
    dataset = load_dataset(args)
    print(f'Loaded video, total frames {dataset.len}')
    if args.n_skip_frames:
        print(f'Skipping {args.n_skip_frames} frames each time, {len(dataset)} left')

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=args.model_cfg.batch_size, pin_memory=True,
        num_workers=args.model_cfg.num_workers, collate_fn=dataset.collate_fn,
        shuffle=False, prefetch_factor=8, # TODO: Test different prefetch_factors
    )

    # Create output writer object
    out_writer = create_out_writer(args, dataset)

    # Load model
    model = load_predictor(args.model_cfg)
    print(f'Infer image size: {args.model_cfg.input_shape}; batch size: {args.model_cfg.batch_size}')

    # Test pre-run
    if args.test:
        model(torch.rand((args.model_cfg.batch_size, 3, *args.model_cfg.input_shape[::-1]), device='cuda'))

    # Infer model
    infer_model(model, dataloader, args, out_writer)

    # Exit from or kill out writer process
    if out_writer is not None:
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
