import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)

def transform(
    m: torch.nn.Module,
    tracer_class: type = torch.fx.Tracer
) -> torch.nn.Module:
    # Note: torch.fx.symbolic_trace is a wrapper around a call to
    # fx.Tracer.trace and constructing a GraphModule.

    # Step 1: acquire a graph representing the code in 'm'
    graph: torch.fx.Graph = tracer_class().trace(m)

    # Step 2: modify this graph or create a new one (custom)
    ...

    # Step 3: construct a Module to return
    return torch.fx.GraphModule(m, graph)

m = MyModule()
gm = transform(m)
print(type(gm))
gm.print_readable()
# <class 'torch.fx.graph_module.GraphModule.__new__.<locals>.GraphModuleImpl'>
# class GraphModule(torch.nn.Module):
#     def forward(self, x : torch.Tensor) -> torch.Tensor:
#         # No stacktrace found for following nodes
#         param = self.param
#         add = x + param;  x = param = None
#         linear = self.linear(add);  add = None
#         clamp = linear.clamp(min = 0.0, max = 1.0);  linear = None
#         return clamp

# It is also possible to modify an existing `GraphModule` instead of creating a new one:
def transform2(m: torch.nn.Module) -> torch.nn.Module:
    gm: torch.fx.GraphModule = torch.fx.symbolic_trace(m)

    # modify gm.graph
    ...

    # recompile the forward() method of `gm` from its graph
    # you must call recompile() to bring the generated `forward()` method
    # on the `GraphModule` in sync with the modified `Graph`
    gm.recompile()

    return gm

gm = transform2(m)
gm.print_readable()
# class MyModule(torch.nn.Module):
#     def forward(self, x : torch.Tensor) -> torch.Tensor:
#         # No stacktrace found for following nodes
#         param = self.param
#         add = x + param;  x = param = None
#         linear = self.linear(add);  add = None
#         clamp = linear.clamp(min = 0.0, max = 1.0);  linear = None
#         return clamp