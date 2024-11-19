# https://github.com/pytorch/examples/blob/main/fx/subgraph_rewriter_basic_use.py

import torch
from torch.fx import symbolic_trace, replace_pattern

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w1, w2):
        val1 = torch.neg(w1)
        m1 = torch.cat([val1, w2]).sum()
        val2 = torch.neg(w1)
        m2 = torch.cat([val2, w2]).sum()
        return x + torch.max(m1) + torch.max(m2)

traced = symbolic_trace(M())
print(traced)
# graph():
#     %x : [num_users=1] = placeholder[target=x]
#     %w1 : [num_users=2] = placeholder[target=w1]
#     %w2 : [num_users=2] = placeholder[target=w2]
#     %neg : [num_users=1] = call_function[target=torch.neg](args = (%w1,), kwargs = {})
#     %cat : [num_users=1] = call_function[target=torch.cat](args = ([%neg, %w2],), kwargs = {})
#     %sum_1 : [num_users=1] = call_method[target=sum](args = (%cat,), kwargs = {})
#     %neg_1 : [num_users=1] = call_function[target=torch.neg](args = (%w1,), kwargs = {})
#     %cat_1 : [num_users=1] = call_function[target=torch.cat](args = ([%neg_1, %w2],), kwargs = {})
#     %sum_2 : [num_users=1] = call_method[target=sum](args = (%cat_1,), kwargs = {})
#     %max_1 : [num_users=1] = call_function[target=torch.max](args = (%sum_1,), kwargs = {})
#     %add : [num_users=1] = call_function[target=operator.add](args = (%x, %max_1), kwargs = {})
#     %max_2 : [num_users=1] = call_function[target=torch.max](args = (%sum_2,), kwargs = {})
#     %add_1 : [num_users=1] = call_function[target=operator.add](args = (%add, %max_2), kwargs = {})
#     return add_1

# define the pattern
# note that pattern-matching is done based on data dependencies, not Node names
def pattern(a1, a2):
    val = torch.neg(a1)
    return torch.cat([val, a2]).sum()

# define the replacement
def replacement(w1, w2):
    return torch.stack([w1, w2])

# replace `pattern` with `replacement` in `traced`
# replace_pattern(gm: GraphModule,
#                 pattern: Callable,
#                 replacement: Callable
#                ) -> None
replace_pattern(traced, pattern, replacement)
print(traced)
# graph():
#     %x : [num_users=1] = placeholder[target=x]
#     %w1 : [num_users=2] = placeholder[target=w1]
#     %w2 : [num_users=2] = placeholder[target=w2]
#     %stack : [num_users=1] = call_function[target=torch.stack](args = ([%w1, %w2],), kwargs = {})
#     %max_1 : [num_users=1] = call_function[target=torch.max](args = (%stack,), kwargs = {})
#     %add : [num_users=1] = call_function[target=operator.add](args = (%x, %max_1), kwargs = {})
#     %stack_1 : [num_users=1] = call_function[target=torch.stack](args = ([%w1, %w2],), kwargs = {})
#     %max_2 : [num_users=1] = call_function[target=torch.max](args = (%stack_1,), kwargs = {})
#     %add_1 : [num_users=1] = call_function[target=operator.add](args = (%add, %max_2), kwargs = {})
#     return add_1