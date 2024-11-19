import torch
from torch.export import export

# accessing tensor data with `.data`
class Bad2(torch.nn.Module):
    def forward(self, x):
        x.data[0, 0] = 3
        return x

print(torch.__version__)
bad2_nonstrict = export(Bad2(), (torch.randn(3, 3), ), strict=False)
print(bad2_nonstrict.module()(torch.ones(3, 3)))

# unsupported functions (such as many built-in functions)
class Bad3(torch.nn.Module):
    def forward(self, x):
        x = x + 1
        return x + id(x)

bad3_nonstrict = export(Bad3(), (torch.randn(3, 3),), strict=False)
print(bad3_nonstrict)
print(bad3_nonstrict.module()(torch.ones(3, 3)))

# unsupported Python language features (e.g. throwing exceptions, match statements)
class Bad4(torch.nn.Module):
    def forward(self, x):
        try:
            x = x + 1
            raise RuntimeError("bad")
        except:
            x = x + 2
        return x

bad4_nonstrict = export(Bad4(), (torch.randn(3, 3),), strict=False)
print(bad4_nonstrict.module()(torch.ones(3, 3)))