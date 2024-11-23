# https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/shape_prop.py
import traceback
from typing import Any, NamedTuple, Optional

import torch
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch._guards import detect_fake_mode
from torch.fx._compatibility import compatibility
from torch.fx.node import map_aggregate, Node

class TensorMetadata(NamedTuple):
    shape: torch.Size
    dtype: torch.dtype
    requires_grad: bool
    stride: tuple[int, ...]

def _extract_tensor_metadata(t: torch.Tensor, including_contiguity=True):
    "Extract a TensorMetadata describing `t`"
    shape = t.shape
    dtype = t.dtype
    requires_grad = t.requires_grad
    stride = t.stride()
    
    return TensorMetadata(
        shape, dtype, requires_grad, stride
    )

@compatibility(is_backward_compatible=True)
class ShapeProp(fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and record the metadata of the result into the corresponding node.
    """
    def __init__(self, gm: fx.GraphModule, fake_mode=None):
        super().__init__(gm)
        if fake_mode is None:
            fake_mode = detect_fake_mode()
        if fake_mode is not None:
            from torch._dynamo.utils import deepcopy_to_fake_tensor

            # Note:
            # We need fake execution cause the inputs are fake, however, we cannot fakity the module
            # because we need to write to the tensor_meta of the real module. So, we fakify to produce
            # a result, to extract tensor meta, and then keep going.
            self.fake_module = deepcopy_to_fake_tensor(self.module, fake_mode)
            self.fake_mode = fake_mode
        else:
            self.fake_module = None
            self.fake_mode = None
        
        self.real_module = self.module
    
    def run_node(self, n: Node) -> Any:
        try:
            if self.fake_module is not None:
                # Hacky swap. Alternatively, we could do this with overriding call_module and get_attr
                self.module = self.fake_module
            try:
                if self.fake_mode is not None:
                    with self.fake_mode, enable_python_dispatcher():
                        result = super().run_node(n)
                else:
                    result = super().run_node(n)
            finally:
                self.module = self.real_module
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"ShapeProp error for: node={n.format_node()} with " f"meta={n.meta}")

        found_tensor = False

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return _extract_tensor_metadata(obj)
            else:
                return obj
        
        meta = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta["tensor_meta"] = meta
        
        n.meta["type"] = type(result)
        return result

    def propagate(self, *args):
        "Run `module` via interpretation and return the result and record the metadata of each node"
        if self.fake_mode is not None:
            fake_args = [
                self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
                for t in args
            ]
        else:
            fake_args = args
        return super().run(*fake_args)

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


model = TwoLayerNet(1000, 100, 10)
gm = torch.fx.symbolic_trace(model)

sample_input = torch.randn(50, 1000)
ShapeProp(gm).propagate(sample_input)

for node in gm.graph.nodes:
    print(node.name, node.meta['tensor_meta'].dtype, node.meta['tensor_meta'].shape)