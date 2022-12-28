import addict

from .base import AbstractWriter
from .multiple import MultipleWriters
from .ffmpeg import FfmpegWriter
from .pillow import PillowWriter


__all__ = [
    'create_writer', 'AbstractWriter', 'FfmpegWriter',
    'PillowWriter', 'MultipleWriters'
]

def create_writer(args, dataset) -> AbstractWriter:
    if args.out_format == 'mp4':
        return FfmpegWriter(cfg=addict.Dict(
            ffmpeg=args.ffmpeg, out_path=args.out_path,
            out_width=args.get('in_width'), out_height=args.get('in_height'),
            out_fps=args.get('in_fps', 30)
        ))
    if args.out_format == 'png':
        return PillowWriter(cfg=addict.Dict(
            in_base_path=args.in_base_path, out_path=args.out_path, ext=args.out_format,
            out_width=args.in_width, out_height=args.in_height, img_num_bits='8bit',
            n_threads=args.n_out_threads,
        ))
    raise ValueError(f'Not supported output format! ({args.out_format})')
