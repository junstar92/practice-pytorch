import torch
from torch.export import Dim, export
import traceback as tb

inp1 = torch.randn(10, 10, 2)

class DynamicShapesExample1(torch.nn.Module):
    def forward(self, x):
        x = x[:, 2:]
        return torch.relu(x)

# inp1_dim0 = Dim("inp1_dim0")  # error occurs in v2.5.1
inp1_dim0 = Dim("inp1_dim0", min=0, max=20)
inp1_dim1 = Dim("inp1_dim1", min=4, max=18)
dynamic_shapes1 = {
    "x": {0: inp1_dim0, 1: inp1_dim1},
}

exported_dynamic_shapes_example1 = export(DynamicShapesExample1(), (inp1,), dynamic_shapes=dynamic_shapes1)

print(exported_dynamic_shapes_example1.module()(torch.randn(5, 5, 2)))

try:
    exported_dynamic_shapes_example1.module()(torch.randn(8, 1, 2))
except Exception:
    tb.print_exc()

try:
    exported_dynamic_shapes_example1.module()(torch.randn(8, 20, 2))
except Exception:
    tb.print_exc()

try:
    exported_dynamic_shapes_example1.module()(torch.randn(8, 8, 3))
except Exception:
    tb.print_exc()