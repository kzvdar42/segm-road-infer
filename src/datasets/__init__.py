import os
from typing import Dict

from src.utils.index import is_image, index_images
from .base_dataset import BaseDataset
from .image_dataset import ImageDataset
from .video_dataset import VideoDataset
from .ffmpeg_dataset import FfmpegVideoDataset

__all__ = ['BaseDataset', 'ImageDataset', 'VideoDataset', 'FfmpegVideoDataset']


def load_dataset(args: Dict) -> BaseDataset:
    """Load dataset based on args."""
    # Create dataloader (if folder/image/txt_file -> image dataset, otherwise video)
    if os.path.isdir(args.in_path) or is_image(args.in_path) or args.in_path.endswith('.txt'):
        image_paths = index_images(args)
        return ImageDataset(image_paths, args, return_raw_imgs=args.show_or_save_mask)
    
    args.base_path = os.path.abspath(os.path.split(args.in_path)[0])
    if args.skip_processed:
        raise NotImplementedError('skip_processed flag is not yet supported for video inputs!')
    if args.n_skip_frames:
        return VideoDataset(
            args.in_path, args, args.n_skip_frames, return_raw_imgs=args.show_or_save_mask
        )
    args.model_cfg.num_workers = 1
    return FfmpegVideoDataset(
        args.in_path, args, args.n_skip_frames, return_raw_imgs=args.show_or_save_mask
    )
