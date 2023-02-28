import os
import json

from addict import Dict
import torch

from .base_predictor import BasePredictor

class OpenVinoPredictor(BasePredictor):

    @staticmethod
    def setup_cfg(cfg_path):
        with open(cfg_path) as in_file:
            config = Dict(json.load(in_file))
            config.model_xml = os.path.join(
                os.path.split(cfg_path)[0],
                config.model_xml
            )
        return config

    def __init__(self, config):
        super().__init__(config)
        from openvino.inference_engine import IECore
        self.ie = IECore()
        self.ie.set_config(
            {
                'CPU_THREADS_NUM': str(config.threads_num),
                'CPU_BIND_THREAD': 'NO'
            },
            config.device,
        )
        self.model_xml = config.weights_path
        self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin"
        self.net = self.ie.read_network(model=self.model_xml, weights=self.model_bin)
        self.input_blob = list(self.net.input_info.keys())[0]
        self.net.reshape(
            {self.input_blob: (config.batch_size, 3, config.input_shape[1], config.input_shape[0])}
        )
        self.net.batch_size = config.batch_size
        self.output_name = self.cfg.get('output_name', next(iter(self.net.outputs)))
        self.exec_net = self.ie.load_network(
            network=self.net,
            num_requests=1,
            device_name= config.device
        )

    def __call__(self, image):
        results = self.exec_net.infer(inputs={self.input_blob: image.cpu().numpy()})
        if self.output_name is None:
            return torch.Tensor(results).cuda()
        elif isinstance(self.output_name, list):
            return (results[out_n] for out_n in self.output_name)
        else:
            return torch.Tensor(results[self.output_name][:, 0]).cuda()
