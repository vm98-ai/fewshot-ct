from .protonet import ProtoNet
from .tim import TIM_GD
from .paddle_mdl import PADDLE
from .laplacian_shot import LaplacianShot

METHOD_REGISTRY = {
    "protonet": ProtoNet,
    "tim": TIM_GD,
    "paddle": PADDLE,
    "laplacianshot": LaplacianShot,
}
