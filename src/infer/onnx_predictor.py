import numpy as np
import torch
import onnxruntime as ort

from src.infer.base_predictor import BasePredictor

class ONNXPredictor(BasePredictor):
    """Class for ONNX based predictor."""

    def __init__(self, cfg):
        super().__init__(cfg)
        # Set graph optimization level
        so = ort.SessionOptions()
        # so.execution_mode = ort.ExecutionMode.ORT_PARALLEL#ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            cfg.providers[cfg.providers.index('CUDAExecutionProvider')] = ("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})
        except ValueError:
            pass

        # Load model
        self.ort_session = ort.InferenceSession(
            cfg.weights_path, sess_options=so, providers=cfg.providers,
        )
        ort_inputs = self.ort_session.get_inputs()
        ort_outputs = self.ort_session.get_outputs()
        assert len(ort_inputs) == 1 and len(ort_outputs) == 1, "Support only models with one input/output"
        self.input_shape = ort_inputs[0].shape
        self.input_name = ort_inputs[0].name
        self.output_shape = ort_outputs[0].shape
        self.output_name = ort_outputs[0].name
        self.io_binding = self.ort_session.io_binding()
        self.seg_output = None

    def __call__(self, in_batch):
        # Create output tensor
        if self.seg_output is None or self.seg_output.shape[1] != in_batch.shape[0]:
            self.seg_output = torch.empty(
                (1, in_batch.shape[0], in_batch.shape[-2], in_batch.shape[-1]),
                dtype=torch.int64, device='cuda'
            )
            self.io_binding.bind_output(
                name=self.output_name, device_type='cuda', device_id=0,
                element_type=np.int64, shape=tuple(self.seg_output.shape),
                buffer_ptr=self.seg_output.data_ptr()
            )

        self.io_binding.bind_input(
            name=self.input_name, device_type='cuda', device_id=0,
            element_type=np.float32, shape=tuple(in_batch.shape),
            buffer_ptr=in_batch.data_ptr()
        )
        torch.cuda.synchronize() # sync for non_blocking data transfer
        self.ort_session.run_with_iobinding(self.io_binding)
        return self.seg_output[0]
