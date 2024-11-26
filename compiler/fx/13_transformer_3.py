import torch
from torch import fx
from torch.fx.node import Target, Argument
from typing import Any
# sample module
class M(torch.nn.Module):
    def forward(self, x, y):
        return torch.mul(x, y) + 1

def args_map(
    target: Target,
    fn,
    args: tuple[Argument, ...],
    kwargs: dict[str, Any],
):
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    args = list(args)
    kwargs = kwargs.copy()

    # update the argument based on the function passed
    def update(key, args, schema):
        args[key] = fn(args[key], schema)
    
    # update each argument in the schema
    for i, schema in enumerate(target._schema.arguments):
        if schema.name in kwargs:
            update(schema.name, kwargs, schema)
        elif not schema.kwarg_only and i < len(args):
            update(i, args, schema)
    
    return tuple(args), kwargs

class ScalarToTensorPass(fx.Transformer):
    def call_function(self, target, args, kwargs):
        def try_coerce(value, arg):
            return (
                torch.tensor(value) if isinstance(value, (float, int, bool)) and type(arg.type) == torch.TensorType
                else value
            )

        args, kwargs = args_map(target, try_coerce, args, kwargs)
        return super().call_function(target, args, kwargs)

m = M()
ep = torch.export.export(m, (torch.randn(5, 5), torch.randn(5, 5), ))
print(ep.graph_module.graph)
# graph():
#     %x : [num_users=1] = placeholder[target=x]
#     %y : [num_users=1] = placeholder[target=y]
#     %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%x, %y), kwargs = {})
#     %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, 1), kwargs = {})
#     return (add,)

transformed_graph_module = ScalarToTensorPass(ep.graph_module).transform()
print(transformed_graph_module.graph)
# graph():
#     %x : [num_users=1] = placeholder[target=x]
#     %y : [num_users=1] = placeholder[target=y]
#     %mul_tensor : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%x, %y), kwargs = {})
#     %_tensor_constant0 : [num_users=1] = get_attr[target=_tensor_constant0]
#     %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_tensor, %_tensor_constant0), kwargs = {})
#     return (add_tensor,)