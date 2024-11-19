import torch
from torch.export import export

# data-dependent control flow
class Bad1(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return torch.sin(x)
        return torch.cos(x)

import traceback as tb
try:
    export(Bad1(), (torch.randn(3, 3),))
except Exception:
    tb.print_exc()

print("=" * 100)

# accessing tensor data with `.data`
class Bad2(torch.nn.Module):
    def forward(self, x):
        x.data[0, 0] = 3
        return x

try:
    export(Bad2(), (torch.randn(3, 3),))
except Exception:
    tb.print_exc()

print("=" * 100)

# calling unsupported functions (such as many built-in functions)
class Bad3(torch.nn.Module):
    def forward(self, x):
        x = x + 1
        return x + id(x)

try:
    export(Bad3(), (torch.randn(3, 3),))
except Exception:
    tb.print_exc()

print("=" * 100)

# unsupported Python language features (e.g. throwing exceptions, match statements)
class Bad4(torch.nn.Module):
    def forward(self, x):
        try:
            x = x + 1
            raise RuntimeError("bad")
        except:
            x = x + 2
        return x

try:
    export(Bad4(), (torch.randn(3, 3),))
except Exception:
    tb.print_exc()