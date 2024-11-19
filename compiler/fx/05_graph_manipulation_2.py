import torch
import torch.fx

# sample module
class M(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)

# add torch.relu() calls node after torch.add() calls node
def transform(
    m: torch.nn.Module,
    tracer_class: type = torch.fx.Tracer
) -> torch.nn.Module:
    graph: torch.fx.Graph = tracer_class().trace(m)
    # FX represents its Graph as an ordered list of nodes
    for node in graph.nodes:
        # check if we're calling a function
        if node.op == 'call_function':
            # the target attribute is the function that call_function calls
            if node.target == torch.add:
                with graph.inserting_after(node):
                    # insert a new `call_function` node calling `torch.relu`
                    new_node = graph.call_function(torch.relu, args=(node, ))
                    # we want all places that used the value of `node` to
                    # now use that value after the `relu` call we've added.
                    node.replace_all_uses_with(new_node)
                    # `replace_all_uses_with` replaces the args of the `new_node`
                    # so, update the args of the `new_node` again
                    new_node.update_arg(0, node)

    # does some checks to make sure the Graph is well-formed
    graph.lint()

    return torch.fx.GraphModule(m, graph)

m = M()
gm = transform(m)
gm.print_readable()
# class GraphModule(torch.nn.Module):
#     def forward(self, x, y):
#         # No stacktrace found for following nodes
#         add = torch.add(x, y);  x = y = None
#         relu = torch.relu(add);  add = None
#         return relu
print(gm.graph)
# graph():
#     %x : [num_users=1] = placeholder[target=x]
#     %y : [num_users=1] = placeholder[target=y]
#     %add : [num_users=1] = call_function[target=torch.add](args = (%x, %y), kwargs = {})
#     %relu : [num_users=1] = call_function[target=torch.relu](args = (%add,), kwargs = {})
#     return relu