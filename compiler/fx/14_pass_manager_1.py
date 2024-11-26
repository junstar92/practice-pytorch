import torch
from torch import fx
from torch.fx.passes.infra.pass_manager import PassManager
from torch.fx.passes.infra.pass_base import PassBase, PassResult

# sample module
class M(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)

# class ReplaceAddWithMul(PassBase):
#     def call(self, graph_module: fx.GraphModule) -> PassResult:
#         nodes = graph_module.graph.nodes
#         for node in nodes:
#             if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
#                 node.target = torch.ops.aten.mul.Tensor
#         return PassResult(graph_module, True)

def replace_add_with_div(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
            node.target = torch.ops.aten.div.Tensor
    return PassResult(gm, True)

def replace_div_with_mul(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.div.Tensor:
            node.target = torch.ops.aten.mul.Tensor
    return PassResult(gm, True)

pass_manager = PassManager(
    passes=[replace_add_with_div, replace_div_with_mul],
    run_checks_after_each_pass=True,
    suppress_check_failures=False
)
# pass_manager.add_pass(ReplaceAddWithMul)

m = M()
ep = torch.export.export(m, (torch.randn(5, 5), torch.randn(5, 5), ))
print(ep.graph_module.graph)
# graph():
#     %x : [num_users=1] = placeholder[target=x]
#     %y : [num_users=1] = placeholder[target=y]
#     %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %y), kwargs = {})
#     return (add,)

pass_result = pass_manager(ep.graph_module)
print(pass_result.graph_module.graph)
# graph():
#     %x : [num_users=1] = placeholder[target=x]
#     %y : [num_users=1] = placeholder[target=y]
#     %add : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%x, %y), kwargs = {})
#     return (add,)