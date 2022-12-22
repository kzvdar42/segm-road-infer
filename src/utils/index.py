import os

from tqdm.auto import tqdm

from .path import get_subfolders_with_files, is_image, get_out_path

def index_ego_masks(root_path: str) -> dict[str, str]:
    img_name_to_ego_mask_paths = {}
    for ego_mask_path in get_subfolders_with_files(root_path, is_image, True):
        rel_ego_mask_name = os.path.splitext(os.path.relpath(ego_mask_path, root_path))[0]
        if rel_ego_mask_name.isdigit():
            rel_ego_mask_name = int(rel_ego_mask_name)
        img_name_to_ego_mask_paths[rel_ego_mask_name] = ego_mask_path
    print(f'Indexed ego masks, found {len(img_name_to_ego_mask_paths)}')
    return img_name_to_ego_mask_paths


def index_images_from_folder(input_folder: str, n_skip_frames: int = 0) -> list[str]:
    # Index images
    print('Indexing images...')
    image_paths = list(get_subfolders_with_files(input_folder, is_image, True))
    print(f'Found {len(image_paths)} images!')
    if n_skip_frames > 0:
        image_paths_ = image_paths
        image_paths = image_paths[::n_skip_frames]
        # Keep last image
        if image_paths_[-1] not in image_paths:
            image_paths.append(image_paths_[-1])
        print(f'Skipping {n_skip_frames} images each time, {len(image_paths)} left')
    return image_paths


def index_images(args) -> list[str]:
    """Index image files from folder/image/txt_file."""
    image_paths = []
    if args.in_path.endswith('.txt'):
        # First line is the base_path for input images!
        with open(args.in_path) as in_stream:
            args.base_path = in_stream.readline().strip()
            image_paths = [l.strip() for l in in_stream.readlines()]
        print(f'Got {len(image_paths)} from input file.')
    else:
        args.base_path = os.path.abspath(args.in_path)
        image_paths = index_images_from_folder(args.in_path, args.n_skip_frames)
    if args.skip_processed:
        pbar = tqdm(image_paths, desc='Check if processed', leave=False)
        image_paths = []
        for img_path in pbar:
            if os.path.isfile(get_out_path(img_path, args.out_path, args.base_path, ext='jpg')):
                image_paths.append(img_path)
        print(f'Skipped already processed files, {len(image_paths)} left')
    return image_paths
