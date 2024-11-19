# https://github.com/pytorch/examples/blob/main/fx/replace_op.py

import torch
from torch.fx import symbolic_trace
import operator

class M(torch.nn.Module):
    def forward(self, x, y):
        return x + y, torch.add(x, y), x.add(y)


traced = symbolic_trace(M())
print(traced.graph)
# graph():
#     %x : [num_users=3] = placeholder[target=x]
#     %y : [num_users=3] = placeholder[target=y]
#     %add : [num_users=1] = call_function[target=operator.add](args = (%x, %y), kwargs = {})
#     %add_1 : [num_users=1] = call_function[target=torch.add](args = (%x, %y), kwargs = {})
#     %add_2 : [num_users=1] = call_method[target=add](args = (%x, %y), kwargs = {})
#     return (add, add_1, add_2)

# There are several different ways to denote addition:
# 1. `x + y` : A `call_function` Node with target `operator.add`
# 2. `torch.add(x, y)` : A `call_funcction` Node with target `torch.add`
# 3. `x.add(y)` : The tensor method call, whose target we can match as a string
patterns = set([operator.add, torch.add, "add"])

# go through all the nodes in the Graph
for n in traced.graph.nodes:
    # if the target matches one of the pattervns
    if any(n.target == pattern for pattern in patterns):
        # set the insert point, add the new node, and replace all uses
        # of `n` with the new node
        with traced.graph.inserting_after(n):
            new_node = traced.graph.call_function(torch.bitwise_and, n.args, n.kwargs)
            n.replace_all_uses_with(new_node)
        # remove the old node from the graph
        traced.graph.erase_node(n)

traced.recompile()
print(traced.graph)
# graph():
#     %x : [num_users=3] = placeholder[target=x]
#     %y : [num_users=3] = placeholder[target=y]
#     %bitwise_and : [num_users=1] = call_function[target=torch.bitwise_and](args = (%x, %y), kwargs = {})
#     %bitwise_and_1 : [num_users=1] = call_function[target=torch.bitwise_and](args = (%x, %y), kwargs = {})
#     %bitwise_and_2 : [num_users=1] = call_function[target=torch.bitwise_and](args = (%x, %y), kwargs = {})
#     return (bitwise_and, bitwise_and_1, bitwise_and_2)