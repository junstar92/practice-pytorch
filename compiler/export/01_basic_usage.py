import torch
from torch.export import export

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x, y):
        return torch.nn.functional.relu(self.lin(x + y), inplace=True)

m = MyModule()
exported_program = export(m, (torch.randn(8, 100), torch.randn(8, 100)))
print(type(exported_program)) # <class 'torch.export.exported_program.ExportedProgram'>
print(exported_program.module()(torch.randn(8, 100), torch.randn(8, 100)))