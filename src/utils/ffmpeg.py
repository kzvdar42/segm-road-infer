import os

import ffmpeg


def ffmpeg_start_in_process(ffmpeg_args, in_filename, scale, codec_name=None):
    vcodec = {
        'hevc': 'hevc_cuvid',
        'h264': 'h264_cuvid',
    }.get(codec_name, None)
    return (
        ffmpeg
        .input(in_filename, vcodec=vcodec)
        .video
        .filter('scale', scale[0], scale[1])
        .filter('setsar', '1')
        .output('pipe:', format='rawvideo', pix_fmt='yuv420p', vsync='vfr')
        .global_args(*ffmpeg_args.in_global_args.split(' ') if ffmpeg_args.in_global_args else [])
        .run_async(pipe_stdout=True)
    )


def ffmpeg_start_out_process(ffmpeg_args, out_filename, in_width, in_height, out_width, out_height, fps=30):
    return (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='gray', s='{}x{}'.format(in_width, in_height), framerate=fps) # hwaccel='cuvid', hwaccel_output_format='cuda', hwaccel='auto'
        .filter('scale', out_width, out_height, sws_flags='neighbor') # neighbor
        # .filter('tmedian', radius=2)
        .output(
            out_filename,
            vcodec=ffmpeg_args.out_vcodec,
            pix_fmt=ffmpeg_args.out_pix_fmt,
            **ffmpeg_args.output_args,
        )
        .global_args(*ffmpeg_args.out_global_args.split(' ') if ffmpeg_args.out_global_args else [])
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

def ffmpeg_start_out_imgs_process(ffmpeg_args, out_path, out_format, in_width, in_height): # , out_width, out_height
    return (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='gray', s='{}x{}'.format(in_width, in_height))
        # .filter('scale', out_width, out_height, flags='neighbor')
        .output(
            os.path.join(out_path, f'%05d.{out_format}'),
            pix_fmt='gray',
        )
        .global_args(*ffmpeg_args.out_global_args.split(' ') if ffmpeg_args.out_global_args else [])
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
