import os
from typing import Callable, Generator, List, Dict

from tqdm.auto import tqdm


def create_folder_for_file(file_path: str) -> None:
    folder_path = os.path.split(file_path)[0]
    if folder_path:
        os.makedirs(folder_path, exist_ok=True)


def is_image(path: str) -> bool:
    path = path.lower()
    for ext in ['jpg', 'jpeg', 'png']:
        if path.endswith(ext):
            return True
    return False


def get_subfolders_with_files(path: str, is_file_func: Callable, yield_by_one: bool = False) -> Generator:
    pbar = tqdm(os.walk(path), leave=False, desc='Indexing...')
    found_files = 0
    for dp, _, fn in pbar:
        file_paths = [os.path.join(dp, f) for f in fn if is_file_func(f)]
        if file_paths:
            if yield_by_one:
                for file_path in file_paths:
                    yield file_path
            else:
                yield file_paths
        found_files += len(file_paths)
        pbar.set_postfix(dict(found_files=found_files))
    pbar.close()


def get_out_path(in_path: str, out_path: str, in_base_path: str, ext: str) -> str:
    rel_img_path = os.path.relpath(in_path, in_base_path)
    # Save masks as png
    rel_img_path = os.path.splitext(rel_img_path)[0] + f'.{ext}'
    pred_out_path = os.path.join(out_path, rel_img_path)
    return pred_out_path


def index_ego_masks(root_path: str) -> Dict[str, str]:
    img_name_to_ego_mask_paths = {}
    for ego_mask_path in get_subfolders_with_files(root_path, is_image, True):
        rel_ego_mask_name = os.path.splitext(os.path.relpath(ego_mask_path, root_path))[0]
        if rel_ego_mask_name.isdigit():
            rel_ego_mask_name = int(rel_ego_mask_name)
        img_name_to_ego_mask_paths[rel_ego_mask_name] = ego_mask_path
    print(f'Indexed ego masks, found {len(img_name_to_ego_mask_paths)}')
    return img_name_to_ego_mask_paths

def index_images_from_folder(input_folder: str, n_skip_frames: int = 0) -> List[str]:
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


def index_images(args) -> List[str]:
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
