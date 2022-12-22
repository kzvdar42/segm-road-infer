from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


def fuse_all_conv_bn(model):
    """Fuse all consecutive Conv2d & BatchNorm2d layer pairs.
    """
    prev_module = (None, None)
    for name, module in model.named_children():
        if list(module.named_children()):
            fuse_all_conv_bn(module)

        if isinstance(module, nn.BatchNorm2d):
            if isinstance(prev_module[1], nn.Conv2d):
                setattr(model, prev_module[0], fuse_conv_bn_eval(prev_module[1], module))
                setattr(model, name, nn.Identity())
        else:
            prev_module = (name, module)
