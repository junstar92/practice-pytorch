import torch
import torch.fx

# sample module
class M(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)

# replace torch.add() calls to torch.mul() calls directly
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
                node.target = torch.mul
    
    # does some checks to make sure the Graph is well-formed
    graph.lint()

    return torch.fx.GraphModule(m, graph)

m = M()
gm = transform(m)
gm.print_readable()
# class GraphModule(torch.nn.Module):
#     def forward(self, x, y):
#         # No stacktrace found for following nodes
#         add = torch.mul(x, y);  x = y = None
#         return add