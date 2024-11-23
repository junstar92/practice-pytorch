import torch
from torch.nn import functional as F
from torch import fx

# note that this decomposition rule can be read as regular Python
def relu_decomposition(x):
    return (x > 0) * x

decomposition_rules = {}
decomposition_rules[F.relu] = relu_decomposition

def decompose(
    model: torch.nn.Module,
    tracer_class: type = fx.Tracer
) -> torch.nn.Module:
    graph: fx.Graph = tracer_class().trace(model)
    print('- before:')
    print(graph)
    # graph():
    #     %x : [num_users=1] = placeholder[target=x]
    #     %relu : [num_users=1] = call_function[target=torch.nn.functional.relu](args = (%x,), kwargs = {inplace: False})
    #     return relu
    new_graph = fx.Graph()
    env = {}
    tracer = fx.proxy.GraphAppendingTracer(new_graph)
    for node in graph.nodes:
        if node.op == 'call_function' and node.target in decomposition_rules:
            # dispatch to the appropriate decomposition rule and
            # implicitly add it to the Graph by symbolically tracing it
            # by warpping the arguments with proxies.
            proxy_args = [
                fx.Proxy(env[x.name], tracer) if isinstance(x, fx.Node) else x for x in node.args
            ]
            output_proxy = decomposition_rules[node.target](*proxy_args)
            new_node = output_proxy.node
            env[node.name] = new_node
        else:
            # default case: just copy the node over into the new graph
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
    
    return fx.GraphModule(model, new_graph)

class M(torch.nn.Module):
    def forward(self, x):
        return F.relu(x)

gm = decompose(M())
print('\n- after:')
print(gm.graph)
# graph():
#     %x : [num_users=2] = placeholder[target=x]
#     %gt : [num_users=1] = call_function[target=operator.gt](args = (%x, 0), kwargs = {})
#     %mul : [num_users=1] = call_function[target=operator.mul](args = (%gt, %x), kwargs = {})
#     return mul