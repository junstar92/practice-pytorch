import torch
from torch.export import Dim, export
import traceback as tb

inp4 = torch.randn(8, 16)
inp5 = torch.randn(16, 32)

class DynamicShapesExample3(torch.nn.Module):
    def forward(self, x, y):
        if x.shape[0] <= 16:
            return x @ y[:, :16]
        return y

dynamic_shapes3 = {
    "x": {i: Dim(f"inp4_dim{i}") for i in range(inp4.dim())},
    "y": {i: Dim(f"inp5_dim{i}") for i in range(inp5.dim())},
}

try:
    export(DynamicShapesExample3(), (inp4, inp5), dynamic_shapes=dynamic_shapes3)
except Exception:
    tb.print_exc()