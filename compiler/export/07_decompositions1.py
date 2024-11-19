import torch
from torch.export import export

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
    
    def forward(self, x):
        return self.linear(x)

ep = export(M(), (torch.randn(2 ,3), ))
print(ep.graph)
# graph():
#     %p_linear_weight : [num_users=1] = placeholder[target=p_linear_weight]
#     %p_linear_bias : [num_users=1] = placeholder[target=p_linear_bias]
#     %x : [num_users=1] = placeholder[target=x]
#     %linear : [num_users=1] = call_function[target=torch.ops.aten.linear.default](args = (%x, %p_linear_weight, %p_linear_bias), kwargs = {})
#     return (linear,)

core_ir_ep = ep.run_decompositions()
print(core_ir_ep.graph)
# graph():
#     %p_linear_weight : [num_users=1] = placeholder[target=p_linear_weight]
#     %p_linear_bias : [num_users=1] = placeholder[target=p_linear_bias]
#     %x : [num_users=1] = placeholder[target=x]
#     %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%p_linear_weight, [1, 0]), kwargs = {})
#     %addmm : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%p_linear_bias, %x, %permute), kwargs = {})
#     return (addmm,)