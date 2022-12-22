import cv2
import numpy as np
import torch

device = 'cpu'

rgb_from_yuv_mat = torch.tensor([
    [1.164,  0,      1.596],
    [1.164, -0.392, -0.813],
    [1.164,  2.017,  0    ],
], device=device).T
rgb_from_yuv_off = torch.tensor([[[16, 128, 128]]], device=device)

def yuv2rgb(image: torch.Tensor) -> torch.Tensor:
    image -= rgb_from_yuv_off
    image @= rgb_from_yuv_mat
    return torch.clamp(image, 0, 255)

def decode_to_torch(in_bytes: bytes, height: int, width: int, device: str, out_dtype=torch.float32) -> torch.Tensor:
    """Decode YUV420p bytes to RGB."""
    k = width*height
    y = torch.empty(k,    dtype=torch.uint8).set_(torch.ByteStorage.from_buffer(in_bytes[0:k],        byte_order = 'native')).reshape((height, width)).type(out_dtype).to(device)
    u = torch.empty(k//4, dtype=torch.uint8).set_(torch.ByteStorage.from_buffer(in_bytes[k:k+(k//4)], byte_order = 'native')).reshape((height//2, width//2)).type(out_dtype).to(device)
    v = torch.empty(k//4, dtype=torch.uint8).set_(torch.ByteStorage.from_buffer(in_bytes[k+(k//4):],  byte_order = 'native')).reshape((height//2, width//2)).type(out_dtype).to(device)
    u = u.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
    v = v.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)

    return yuv2rgb(torch.stack((y,u,v), -1))

def decode_to_numpy(in_bytes: bytes, height: int, width: int) -> np.ndarray:
    """Decode YUV420p bytes to RGB."""
    return cv2.cvtColor(
        np.frombuffer(in_bytes, dtype=np.uint8).reshape((height + height//2, width)),
        cv2.COLOR_YUV420p2RGB
    )
