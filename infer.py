import argparse
import os
from typing import List, Tuple

import addict
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils import get_subfolders_with_files, is_image, load_model_config, colorize_pred, apply_mask, create_folder_for_file, get_classes
from infer_utils import Predictor, ONNXPredictor, DetectronPredictor, ImageDataset


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


def save_preds(predictions: List[np.ndarray], metadata: dict, in_base_path: str, out_path: str) -> None:
    for pred, meta in zip(predictions, metadata):
        rel_img_path = os.path.relpath(meta['image_path'], in_base_path)
        # Save masks as png
        rel_img_path = os.path.splitext(rel_img_path)[0] + '.png'
        pred_out_path = os.path.join(out_path, rel_img_path)
        create_folder_for_file(pred_out_path)
        if not cv2.imwrite(pred_out_path, pred):
            print(f'Didn\'t save!', pred_out_path, type(pred), pred.shape)

def get_args():
    parser = argparse.ArgumentParser('Segmentation Inference Script')
    parser.add_argument('model_config', help='path to the model yaml config')
    parser.add_argument('in_path', help='path to input images. Will read all images under this path')
    parser.add_argument('out_path', help='path to save the resulting masks')
    parser.add_argument('--show', action='store_true', help='set to visualize predictions')
    parser.add_argument('--apply_ego_mask_from', help='path to ego masks, will load them and apply to predictions')
    parser.add_argument('--n_skip_frames', type=int, default=0, help='how many frames to skip during inference [default: 0]')
    parser.add_argument('--only_ego_vehicle', action='store_true', help='store only ego vehicle class')
    parser.add_argument('--save_vis_to', default=None, help='path to save the visualized predictions. [default: None]')
    parser.add_argument('--window_size', type=int, nargs='+', default=(1280, 720), help='window size for visualization')

    args = parser.parse_args()
    args = addict.Dict(vars(args))
    return args


if __name__ == '__main__':
    # Get args
    args = get_args()
    print('Args:', args, sep='\n')

    # Load configs
    model_cfg = load_model_config(args.model_config)

    # Load classes
    classes, cls_name_to_id, cls_id_to_name = get_classes(model_cfg.classes)
    assert not(args.only_ego_vehicle) or 'ego vehicle' in classes, \
        'Model without ego vehicle class!'

    # Index ego masks if provided
    img_name_to_ego_mask_paths = {}
    ego_mask = None
    if args.apply_ego_mask_from:
        for ego_mask_path in get_subfolders_with_files(args.apply_ego_mask_from, is_image, True):
            rel_ego_mask_path = os.path.relpath(ego_mask_path, args.apply_ego_mask_from)
            img_name_to_ego_mask_paths[rel_ego_mask_path] = ego_mask_path
            # Load first mask
            if ego_mask is None:
                ego_mask = cv2.imread(ego_mask_path, -1)
        print(f'Indexed ego mask, found {len(img_name_to_ego_mask_paths)}')


    # Index images
    print('Indexing files...')
    image_paths = get_subfolders_with_files(args.in_path, is_image, True)
    image_paths = list(image_paths)
    print(f'Found {len(image_paths)} images!')
    if args.n_skip_frames > 0:
        image_paths_ = image_paths
        image_paths = image_paths[::args.n_skip_frames]
        # Keep last image
        if image_paths_[-1] not in image_paths:
            image_paths.append(image_paths_[-1])
        print(f'Skipping {args.n_skip_frames} images each time, {len(image_paths)} left')

    # Create dataloader
    dataset = ImageDataset(image_paths, model_cfg)
    dataloader = DataLoader(
        dataset, batch_size=model_cfg.batch_size, pin_memory=True,
        num_workers=model_cfg.batch_size, collate_fn=dataset.collate_fn,
    )

    # Load model
    model = load_predictor(model_cfg)

    # infer model
    pbar = tqdm(dataloader)
    for images, metadata in pbar:
        predictions = model(images)
        processed = dataset.postprocess(predictions, metadata)
        del predictions

        if args.only_ego_vehicle:
            for pred_num, pred in enumerate(processed):
                only_ego_vehicle = np.zeros_like(pred, dtype=bool)
                for cls_name in ['ego vehicle', 'car mount']:
                    only_ego_vehicle = np.logical_or(only_ego_vehicle, pred == cls_name_to_id[cls_name])
                processed[pred_num] = only_ego_vehicle.astype(np.uint8) * 255
        elif len(img_name_to_ego_mask_paths):
            for pred_num, (pred, meta) in enumerate(zip(processed, metadata)):
                ego_mask_rel_path = os.path.relpath(meta['image_path'], args.in_path)
                if ego_mask_rel_path in img_name_to_ego_mask_paths:
                    ego_mask = cv2.imread(img_name_to_ego_mask_paths[ego_mask_rel_path], -1)
                    dataset.img_mask = ego_mask
                if ego_mask is not None:
                    pred[ego_mask == 255] = len(classes)

        if args.show or args.save_vis_to:
            if show_preds(processed, metadata, args.show, args.save_vis_to, args.window_size):
                cv2.destroyAllWindows()
                break  

        if args.out_path:
            save_preds(processed, metadata, args.in_path, args.out_path)
