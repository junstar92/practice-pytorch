import torch
from torch.export import export

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        x = self.linear(x)
        return x

inp = torch.randn(2, 3, device="cuda")
m = M().to(device="cuda")
ep = torch.export.export(m, (inp,))

# Run it eagerly
res = ep.module()(inp)
print(res)

# Run it with torch.compile
res = torch.compile(ep.module(), backend="inductor")(inp)
print(res)