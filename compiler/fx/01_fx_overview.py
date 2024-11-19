import torch

class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)

module = MyModule()

from torch.fx import symbolic_trace
# symbolic tracing frontend - captures the semantics of the module
symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)

# high-level intermediate representation - graph representation
print(symbolic_traced.graph)
# graph():
#     %x : torch.Tensor [num_users=1] = placeholder[target=x]
#     %param : [num_users=1] = get_attr[target=param]
#     %add : [num_users=1] = call_function[target=operator.add](args = (%x, %param), kwargs = {})
#     %linear : [num_users=1] = call_module[target=linear](args = (%add,), kwargs = {})
#     %clamp : [num_users=1] = call_method[target=clamp](args = (%linear,), kwargs = {min: 0.0, max: 1.0})
#     return clamp

# code generation - valid python code
print(symbolic_traced.code)
# def forward(self, x : torch.Tensor) -> torch.Tensor:
#     param = self.param
#     add = x + param;  x = param = None
#     linear = self.linear(add);  add = None
#     clamp = linear.clamp(min = 0.0, max = 1.0);  linear = None
#     return clamp