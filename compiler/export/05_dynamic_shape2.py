import torch
from torch.export import Dim, export
import traceback as tb

class DynamicShapesExample2(torch.nn.Module):
    def forward(self, x, y):
        return x @ y

inp2 = torch.randn(4, 8)
inp3 = torch.randn(8, 2)

# enforces that euqalities between dimensions of different tensors by using the same `torch.exmport.Dim`.
inp2_dim0 = Dim("inp2_dim0")
inner_dim = Dim("inner_dim")
inp3_dim1 = Dim("inp3_dim1")

dynamic_shapes2 = {
    "x": {0: inp2_dim0, 1: inner_dim},
    "y": {0: inner_dim, 1: inp3_dim1},
}

exported_dynamic_shapes_example2 = export(DynamicShapesExample2(), (inp2, inp3), dynamic_shapes=dynamic_shapes2)

print(exported_dynamic_shapes_example2.module()(torch.randn(2, 16), torch.randn(16, 4)))

try:
    exported_dynamic_shapes_example2.module()(torch.randn(4, 8), torch.randn(4, 2))
except Exception:
    tb.print_exc()