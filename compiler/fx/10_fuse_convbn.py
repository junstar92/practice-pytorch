# https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L50

from typing import Any
import copy
import torch
from torchvision.models import resnet18
from torch import nn
from torch.export import export

from torch.fx import GraphModule, Node, symbolic_trace
from torch.fx.passes.infra.pass_base import PassBase, PassResult

def matches_pattern(
    node: Node,
    pattern: tuple[nn.Module, nn.Module],
    modules: dict[str, Any],
) -> bool:
    if not node.all_input_nodes:
        return False
    nodes = (node.all_input_nodes[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if current_node.op != "call_module" or not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules or type(modules[current_node.target]) is not expected_type:
            return False
    return True

def _fuse_convbn(
    conv: torch.nn.Module,
    bn: torch.nn.Module
) -> torch.nn.Module:
    fused_conv = copy.deepcopy(conv)
    conv_w, conv_b = fused_conv.weight, fused_conv.bias
    bn_running_mean, bn_running_var, bn_eps, bn_w, bn_b = bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias
    conv_w_dtype = conv_w.dtype
    conv_b_dtype = conv_b.dtype if conv_b is not None else conv_w_dtype
    if conv_b is None:
        conv_b = torch.zeros_like(bn_running_mean)
    if bn_w is None:
        bn_w = torch.ones_like(bn_running_mean)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_running_mean)
    bn_var_rsqrt = torch.rsqrt(bn_running_var + bn_eps)

    shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)

    fused_conv_w = (conv_w * (bn_w * bn_var_rsqrt).reshape(shape)).to(dtype=conv_w_dtype)
    fused_conv_b = ((conv_b - bn_running_mean) * bn_var_rsqrt * bn_w + bn_b).to(dtype=conv_b_dtype)
    
    fused_conv.weight = torch.nn.Parameter(fused_conv_w, conv_w.requires_grad)
    fused_conv.bias = torch.nn.Parameter(fused_conv_b, conv_w.requires_grad)

    return fused_conv

def fuse_convbn(model: GraphModule) -> None:
    model.eval()
    modules = dict(model.named_modules())
    patterns = [(nn.Conv1d, nn.BatchNorm1d),
                (nn.Conv2d, nn.BatchNorm2d),
                (nn.Conv3d, nn.BatchNorm3d)]
    
    graph = model.graph
    for pattern in patterns:
        for node in graph.nodes:
            if matches_pattern(node, pattern, modules):
                if len(node.all_input_nodes[0].users) > 1: # output of conv is used by other nodes
                    continue
                conv_node = node.all_input_nodes[0]
                bn_node = node
                conv, bn = modules[conv_node.target], modules[bn_node.target]
                fused_conv = _fuse_convbn(conv, bn)
                model.add_submodule(conv_node.target, fused_conv)
                bn_node.replace_all_uses_with(conv_node)
                model.graph.erase_node(bn_node)
                model.delete_submodule(bn_node.target)
    graph.lint()
    model.recompile()

model = resnet18()
graph_module = symbolic_trace(model)
# exported_program = export(model, (torch.randn(1, 3, 224, 224), ))
torch.onnx.export(graph_module, (torch.randn(1, 3, 224, 224), ), 'resnet18_gm.onnx')
# torch.onnx.export(exported_program, (torch.randn(1, 3, 224, 224), ), 'resnet18_ep.onnx', external_data=False)

# fuse conv_bn
fuse_convbn(graph_module)

# torch.onnx.export(model, (torch.randn(1, 3, 224, 224), ), 'resnet18_orig.onnx')
torch.onnx.export(graph_module, (torch.randn(1, 3, 224, 224), ), 'resnet18_fused.onnx')