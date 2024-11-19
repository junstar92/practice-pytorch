import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.topk(torch.sum(
            self.linear(x + self.linear.weight).relu(), dim=-1), 3
        )

m = MyModule()
gm = torch.fx.symbolic_trace(m)

gm.graph.print_tabular()
# opcode         name           target                                                args                kwargs
# -------------  -------------  ----------------------------------------------------  ------------------  -----------
# placeholder    x              x                                                     ()                  {}
# get_attr       linear_weight  linear.weight                                         ()                  {}
# call_function  add            <built-in function add>                               (x, linear_weight)  {}
# call_module    linear         linear                                                (add,)              {}
# call_method    relu           relu                                                  (linear,)           {}
# call_function  sum_1          <built-in method sum of type object at 0x1038e69b8>   (relu,)             {'dim': -1}
# call_function  topk           <built-in method topk of type object at 0x1038e69b8>  (sum_1, 3)          {}
# output         output         output                                                (topk,)             {}