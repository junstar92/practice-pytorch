import torch
from torch import fx

# sample module
class M(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)

class ReplaceAddWithMulSub(fx.Transformer):
    def call_function(self, target, args, kwargs):
        if target != torch.ops.aten.add.Tensor:
            return super().call_function(target, args, kwargs)

        _, y = args
        mul = super().call_function(torch.ops.aten.mul.Tensor, args, {})
        return super().call_function(torch.ops.aten.sub.Tensor, (mul, y), {})

m = M()
ep = torch.export.export(m, (torch.randn(5, 5), torch.randn(5, 5), ))
print(ep.graph_module.graph)
# graph():
#     %x : [num_users=1] = placeholder[target=x]
#     %y : [num_users=1] = placeholder[target=y]
#     %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %y), kwargs = {})
#     return (add,)

transformed_graph_module = ReplaceAddWithMulSub(ep.graph_module).transform()
print(transformed_graph_module.graph)
# graph():
#     %x : [num_users=1] = placeholder[target=x]
#     %y : [num_users=2] = placeholder[target=y]
#     %mul_tensor : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%x, %y), kwargs = {})
#     %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %y), kwargs = {})
#     return (sub_tensor,)