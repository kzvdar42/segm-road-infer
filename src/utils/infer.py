import time

import addict


class PseudoTqdm:
    """Class to use as in-place replacement for tqdm then you don't need a pbar."""

    def __init__(self, print_step: int):
        self.start_t = time.time() 
        self.n_runs = 0
        self.print_step = print_step

    @property
    def rate(self):
        return self.n_runs / (time.time() - self.start_t)

    def update(self, n_runs : int):
        self.n_runs += n_runs
        if self.n_runs % self.print_step == 0:
            print(f'Processed {self.n_runs} images, at rate {self.rate:.2f} imgs/s', flush=True)
    
    def close(self):
        print(f'Processed {self.n_runs} images, at rate {self.rate:.2f} imgs/s. Total: {time.time() - self.start_t:.2f} sec')
        pass

def select_batch_size(args: addict.Dict) -> addict.Dict:
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
    return args
