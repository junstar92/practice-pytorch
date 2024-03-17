"""
Rotary Positional Embedding (RoPE) Implementation for Torch, CUDA, and Triton

References
- https://arxiv.org/pdf/2104.09864.pdf
- https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/rope_embedding.py
- https://github.com/NVIDIA/TransformerEngine/blob/b8eea8aaa94bb566c3a12384eda064bda8ac4fd7/transformer_engine/pytorch/attention.py#L1170
- https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/rotary.py
- https://github.com/facebookresearch/xformers/blob/main/xformers/components/positional_embedding/rotary.py
"""

import random
import numpy as np
from typing import Union, Tuple
import torch
import torch.backends.cudnn as cudnn
import triton
import triton.language as tl
import transformer_engine
import transformer_engine_extensions as tex

class FusedRoPEFunc(torch.autograd.Function):
    """
    Function for FusedRoPE

    This implementation assumes the input tensor to be in `sbhd`, `bshd` or `thd` format and
    the RoPE tensor to be of shape (s, 1, 1, d). It accepts arbitrary memory layouts to avoid
    the expensive `.contiguous()` calls, thus it may not achieve the best memory access pattern.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
        cu_seqlens: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if tensor_format == "sbhd":
            output = tex.fused_rope_forward(t, freqs, False)
        elif tensor_format == "bshd":
            output = tex.fused_rope_forward(
                t.transpose(0, 1), freqs, True
            ).transpose(0, 1)
        elif tensor_format == "thd":
            output = tex.fused_rope_thd_forward(t, cu_seqlens, freqs)
        else:
            raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
        ctx.save_for_backward(freqs, cu_seqlens)
        ctx.tensor_format = tensor_format

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        freqs, cu_seqlens = ctx.saved_tensors
        if ctx.tensor_format == "sbhd":
            grad_input = tex.fused_rope_backward(grad_output, freqs, False)
        elif ctx.tensor_format == "bshd":
            grad_input = tex.fused_rope_backward(
                grad_output.transpose(0, 1), freqs, True
            ).transpose(0, 1)
        elif ctx.tensor_format == "thd":
            grad_input = tex.fused_rope_thd_backward(grad_output, cu_seqlens, freqs)
        else:
            raise ValueError(f"Unsupported tensor_format: {ctx.tensor_format}.")

        return grad_input, None, None, None, None

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
) -> torch.Tensor:
    """
    Apply rotary positional embedding tensor to the input tensor.

    Parameters
    ----------
    t: torch.Tensor
        Input tensor of shape `[s, b, h, d]`, `[s, b, h, d]` or `[t, h, d]`, on which
        rotary positional embedding will be applied.
    freqs: torch.Tensor
        Rotary positional embedding tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    fused: bool, default = False
        Whether to use a fused applying RoPE implementation.
    tensor_format: {'sbhd', 'bshd', 'thd'}, default = 'sbhd'
        is `bshd` if `t` is of shape `[bs, seq, ...]`, or `sbhd` if `t` is
        of shape `[seq, bs, ...]`. 'thd' is only supported when `fused` is True.
    cu_seqlens: torch.Tensor, default = None.
        Cumulative sum of sequence lengths in a batch for `t`, with shape [b + 1] and
        dtype torch.int32. Only valid when `tensor_format` is 'thd'.
    """
    if fused:
        assert (
            tensor_format != "thd" or cu_seqlens is not None
        ), "cu_seqlens must not be None when tensor_format is 'thd'."
        return FusedRoPEFunc.apply(t, freqs, tensor_format, cu_seqlens)

    assert tensor_format in ("sbhd", "bshd"), (
        "Only formats `sbhd` or `bshd` are supported for input tensor `t` "
        f"when fused is False, got {tensor_format}."
    )

    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]

    # Only apply the rotary embeddings up to the sequence length of the running
    # input.
    assert cur_seq_len <= max_seq_len, (
        f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
    )
    freqs = freqs[:cur_seq_len]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)  # [seq, 1, 1, dim] -> [1, seq, 1, dim]
    # cos/sin first then dtype conversion for better precision
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)

def rope_torch(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
):
    out = apply_rotary_pos_emb(t, freqs, tensor_format, fused, cu_seqlens)
    return out

@triton.jit
def rope_fw(
    # pointer to inputs
    t_ptr, freqs_ptr,
    # pointer to output
    out_ptr,
    # dimensions
    seqlen, batch, num_heads, d_model, rotary_dim,
    # stride variables
    stride_t_seqlen, stride_t_batch, stride_t_nheads, stride_t_headdim,
    # meta-params
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    rotary_dim_half = rotary_dim // 2

    freqs = tl.load(freqs_ptr + (pid_m % seqlen) * rotary_dim + col_offsets, mask=col_offsets < rotary_dim_half, other = 0)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)

    t1 = tl.load(t_ptr + (pid_m % seqlen) * stride_t_seqlen + \
                 (pid_m // seqlen) * stride_t_batch + \
                 pid_head * d_model + col_offsets,
                 mask=col_offsets < rotary_dim_half, other=0)
    t2 = tl.load(t_ptr + (pid_m % seqlen) * stride_t_seqlen + \
                 (pid_m // seqlen) * stride_t_batch + \
                 pid_head * d_model + rotary_dim_half + col_offsets,
                 mask=col_offsets < rotary_dim_half, other=0)
    
    tl.store(out_ptr + (pid_m % seqlen) * stride_t_seqlen + \
             (pid_m // seqlen) * stride_t_batch + \
             pid_head * d_model + col_offsets,
             t1 * cos - t2 * sin,
             mask=col_offsets < rotary_dim_half)
    tl.store(out_ptr + (pid_m % seqlen) * stride_t_seqlen + \
             (pid_m // seqlen) * stride_t_batch + \
             pid_head * d_model + rotary_dim_half + col_offsets,
             t2 * cos + t1 * sin,
             mask=col_offsets < rotary_dim_half)

@triton.jit
def rope_bw(
    # pointer to inputs
    t_ptr, freqs_ptr,
    # pointer to output
    out_ptr,
    # dimensions
    seqlen, batch, num_heads, d_model, rotary_dim,
    # stride variables
    stride_t_seqlen, stride_t_batch, stride_t_nheads, stride_t_headdim,
    # meta-params
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    rotary_dim_half = rotary_dim // 2

    freqs = tl.load(freqs_ptr + (pid_m % seqlen) * rotary_dim + col_offsets, mask=col_offsets < rotary_dim_half, other = 0)
    cos = tl.cos(freqs)
    sin = -tl.sin(freqs)

    t1 = tl.load(t_ptr + (pid_m % seqlen) * stride_t_seqlen + \
                 (pid_m // seqlen) * stride_t_batch + \
                 pid_head * d_model + col_offsets,
                 mask=col_offsets < rotary_dim_half, other=0)
    t2 = tl.load(t_ptr + (pid_m % seqlen) * stride_t_seqlen + \
                 (pid_m // seqlen) * stride_t_batch + \
                 pid_head * d_model + rotary_dim_half + col_offsets,
                 mask=col_offsets < rotary_dim_half, other=0)
    
    tl.store(out_ptr + (pid_m % seqlen) * stride_t_seqlen + \
             (pid_m // seqlen) * stride_t_batch + \
             pid_head * d_model + col_offsets,
             t1 * cos - t2 * sin,
             mask=col_offsets < rotary_dim_half)
    tl.store(out_ptr + (pid_m % seqlen) * stride_t_seqlen + \
             (pid_m // seqlen) * stride_t_batch + \
             pid_head * d_model + rotary_dim_half + col_offsets,
             t2 * cos + t1 * sin,
             mask=col_offsets < rotary_dim_half)


def calculate_settings(n):
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps

class RopeTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
    ):        
        seqlen, batch, num_heads, d_model = t.shape
        assert(seqlen <= freqs.shape[0])

        output = torch.empty_like(t)

        BLOCK_SIZE, num_warps = calculate_settings(d_model)
        rope_fw[(seqlen * batch, num_heads,)](
            t, freqs,
            output,
            seqlen, batch, num_heads, d_model, freqs.shape[-1],
            t.stride(0), t.stride(1), t.stride(2), t.stride(3),
            BLOCK_SIZE,
            num_warps=num_warps,
        )
        
        ctx.save_for_backward(freqs)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        
        return output

    @staticmethod
    def backward(
        ctx,
        dY: torch.Tensor,
    ):
        freqs, = ctx.saved_tensors
        seqlen, batch, num_heads, d_model = dY.shape

        output = torch.zeros_like(dY)

        rope_bw[(seqlen * batch, num_heads,)](
            dY, freqs,
            output,
            seqlen, batch, num_heads, d_model, freqs.shape[-1],
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps
        )
        
        return output, None

rope_cuda = FusedRoPEFunc.apply
rope_triton = RopeTriton.apply

def get_freqs(
    max_seqlen: int,
    d_model: int
):
    t = torch.arange(max_seqlen)
    inv_freqs = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)) #torch.ones([(int)(d_model / 2)])
    freqs = torch.einsum('i, j -> i j', t.type_as(inv_freqs), inv_freqs)
    freqs = torch.cat((freqs, freqs), dim=-1).reshape(max_seqlen, 1, 1, d_model)
    return freqs

if __name__ == '__main__':
    # set random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    cudnn.benchmark = False
    cudnn.deterministic = True

    MAX_SEQLEN = 1024
    BATCH_SIZE = 10
    SEQLEN = 256
    NUM_HEADS = 96
    D_MODEL = 256

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    input = torch.randn([SEQLEN, BATCH_SIZE, NUM_HEADS, D_MODEL]).to(device)
    freqs = get_freqs(MAX_SEQLEN, D_MODEL).to(device)

    # torch implementation
    input_torch = input.detach().clone().requires_grad_(True)
    output_torch = rope_torch(input_torch, freqs)
    output_torch.backward(gradient=torch.ones_like(output_torch))
    grad_torch = input_torch.grad

    # triton implementation
    input_triton = input.detach().clone().requires_grad_(True)
    output_triton = rope_triton(input_triton, freqs)
    output_triton.backward(gradient=torch.ones_like(output_triton))
    grad_triton = input_triton.grad

    # assertion
    torch.testing.assert_close(output_torch, output_triton)
    torch.testing.assert_close(grad_torch, grad_triton)

    # benchmark (torch vs cuda vs triton)
    @triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['SEQ_LEN'],
        x_vals=[128 * i for i in range(2, 9)],
        line_arg='provider',
        line_vals=[
            'torch-native',
            'torch-cuda (sbhd)',
            'torch-cuda (tbd)',
            'triton',
        ],
        line_names=[
            "Torch (native)",
            "Torch (cuda-sbhd)",
            "Torch (cuda-tbd)",
            "Triton",
        ],
        styles=[('blue', '-'), ('green', '-'), ('orange', ':'), ('red', '--')],
        ylabel="ms",
        plot_name="rope-performance (batch_size: 10, num_heads: 96, head_dim: 128)", 
        args={
            'BATCH_SIZE': 10,
            'NUM_HEADS': 96,
            'HEAD_DIM': 256,
        },
    ))
    def benchmark(SEQ_LEN, BATCH_SIZE, NUM_HEADS, HEAD_DIM, provider):
        x = torch.randn(SEQ_LEN, BATCH_SIZE, NUM_HEADS, HEAD_DIM, device='cuda', dtype=torch.float32)
        percentiles = [0.5, 0.2, 0.8]
        if provider == 'torch-native':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope_torch(x, freqs), percentiles=percentiles)
        if provider == 'torch-cuda (sbhd)':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope_cuda(x, freqs), percentiles=percentiles)
        if provider == 'torch-cuda (tbd)':
            x = x.reshape(-1, x.shape[-2], x.shape[-1])
            cu_seqlen = torch.Tensor([0] + [(i + 1) * SEQ_LEN for i in range(BATCH_SIZE)]).to(torch.int32).to(device)
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope_cuda(x, freqs, "thd", cu_seqlen), percentiles=percentiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope_triton(x, freqs), percentiles=percentiles)
        return ms, max_ms, min_ms

    benchmark.run(show_plots=True, print_data=True)