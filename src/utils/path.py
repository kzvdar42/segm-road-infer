import os
from typing import Callable, Generator
import yaml

import addict
from tqdm.auto import tqdm


mask2former_home_path = os.environ.get('MASK2FORMER_HOME')


def load_model_config(cfg_path: str) -> addict.Dict:
    """Load config file for the predictor."""
    with open(cfg_path, 'r') as in_stream:
        cfg = yaml.safe_load(in_stream)
        if 'config_file' in cfg.keys():
            if not cfg['config_file'].startswith('/'):
                assert mask2former_home_path is not None, 'Need to set `mask2former_home_path` env var!'
                cfg['config_file'] = os.path.join(mask2former_home_path, cfg['config_file'])
        return addict.Dict(cfg)


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
