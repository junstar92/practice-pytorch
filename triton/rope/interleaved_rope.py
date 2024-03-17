import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

import triton
import triton.language as tl
import transformer_engine
import transformer_engine_extensions as tex

def rotate_interleave(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-2).transpose(-1, -2).reshape(*x.shape[:-1], -1)

def apply_interleaved_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,    
) -> torch.Tensor:
    """
    RoPE Implementation for Interleaved Version

    Parameters
    ----------
    t: torch.Tensor (seqlen, batch_size, num_heads, head_dim)
    freqs: (seqlen, 1, 1, rot_dim)
    """
    cur_seq_len = t.shape[0]
    max_seq_len = freqs.shape[0]
    assert cur_seq_len <= max_seq_len

    freqs = freqs[:cur_seq_len]
    rot_dim = freqs.shape[-1]
    assert rot_dim <= t.shape[-1]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    return torch.cat(
        (t[..., :rot_dim] * cos + rotate_interleave(t[..., :rot_dim]) * sin, t[..., rot_dim:]), dim=-1
    )

def rope_interleaved_torch(
    t: torch.Tensor,
    freqs: torch.Tensor,
):
    out = apply_interleaved_rotary_pos_emb(t, freqs)
    return out

# RoPE Triton Implementation for Interleaved Version
@triton.jit
def rope_interleaved_fw(
    t_ptr, freqs_ptr,
    out_ptr,
    seqlen, batch, num_heads, d_model, rotary_dim,
    stride_t_seqlen, stride_t_batch, stride_t_nheads, stride_t_headdim,
    BLOCK_SIZE: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    col_offsets = tl.arange(0, BLOCK_SIZE // 2)
    rotary_dim_half = rotary_dim // 2

    freqs = tl.load(freqs_ptr + (pid_m % seqlen) * rotary_dim + col_offsets * 2, mask=col_offsets < rotary_dim_half, other = 0)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)

    odd = tl.load(t_ptr + (pid_m % seqlen) * stride_t_seqlen + \
                  (pid_m // seqlen) * stride_t_batch + \
                  pid_head * d_model + col_offsets * 2,
                  mask=col_offsets < rotary_dim_half) # [x_1, x_3, x_5, ..., x_d_1]
    even = tl.load(t_ptr + (pid_m % seqlen) * stride_t_seqlen + \
                   (pid_m // seqlen) * stride_t_batch + \
                   pid_head * d_model + col_offsets * 2 + 1,
                   mask=col_offsets < rotary_dim_half) # [x_2, x_4, x_6 ..., x_d]

    tl.store(out_ptr + (pid_m % seqlen) * stride_t_seqlen + \
             (pid_m // seqlen) * stride_t_batch + \
             pid_head * d_model + col_offsets * 2,
             odd * cos - even * sin,
             mask=col_offsets < rotary_dim_half)
    tl.store(out_ptr + (pid_m % seqlen) * stride_t_seqlen + \
             (pid_m // seqlen) * stride_t_batch + \
             pid_head * d_model + col_offsets * 2 + 1,
             even * cos + odd * sin,
             mask=col_offsets < rotary_dim_half)

@triton.jit
def rope_interleaved_bw(
    t_ptr, freqs_ptr,
    out_ptr,
    seqlen, batch, num_heads, d_model, rotary_dim,
    stride_t_seqlen, stride_t_batch, stride_t_nheads, stride_t_headdim,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    col_offsets = tl.arange(0, BLOCK_SIZE // 2)
    rotary_dim_half = rotary_dim // 2

    freqs = tl.load(freqs_ptr + (pid_m % seqlen) * rotary_dim + col_offsets * 2, mask=col_offsets < rotary_dim_half, other = 0)
    cos = tl.cos(freqs)
    sin = -tl.sin(freqs)

    odd = tl.load(t_ptr + (pid_m % seqlen) * stride_t_seqlen + \
                  (pid_m // seqlen) * stride_t_batch + \
                  pid_head * d_model + col_offsets * 2,
                  mask=col_offsets < rotary_dim_half) # [x_1, x_3, x_5, ..., x_d_1]
    even = tl.load(t_ptr + (pid_m % seqlen) * stride_t_seqlen + \
                   (pid_m // seqlen) * stride_t_batch + \
                   pid_head * d_model + col_offsets * 2 + 1,
                   mask=col_offsets < rotary_dim_half) # [x_2, x_4, x_6 ..., x_d]

    tl.store(out_ptr + (pid_m % seqlen) * stride_t_seqlen + \
             (pid_m // seqlen) * stride_t_batch + \
             pid_head * d_model + col_offsets * 2,
             odd * cos - even * sin,
             mask=col_offsets < rotary_dim_half)
    tl.store(out_ptr + (pid_m % seqlen) * stride_t_seqlen + \
             (pid_m // seqlen) * stride_t_batch + \
             pid_head * d_model + col_offsets * 2 + 1,
             even * cos + odd * sin,
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

class RopeInterleavedTriton(torch.autograd.Function):
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
        rope_interleaved_fw[(seqlen * batch, num_heads,)](
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

        rope_interleaved_bw[(seqlen * batch, num_heads,)](
            dY, freqs,
            output,
            seqlen, batch, num_heads, d_model, freqs.shape[-1],
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps
        )
        
        return output, None

rope_interleaved_triton = RopeInterleavedTriton.apply

# Strided RoPE Implementation
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
) -> torch.Tensor:
    """
    Apply rotary positional embedding tensor to the input tensor.

    Parameters
    ----------
    t: torch.Tensor
        Input tensor of shape `[s, b, h, d]`, on which
        rotary positional embedding will be applied.
    freqs: torch.Tensor
        Rotary positional embedding tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    """

    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[0]

    # Only apply the rotary embeddings up to the sequence length of the running
    # input.
    assert cur_seq_len <= max_seq_len, (
        f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
    )
    freqs = freqs[:cur_seq_len]
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)

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

rope_strided_torch = apply_rotary_pos_emb
rope_strided_triton = RopeTriton.apply

def get_freqs(
    max_seqlen: int,
    d_model: int
):
    t = torch.arange(max_seqlen)
    inv_freqs = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
    freqs = torch.einsum('i,j -> ij', t.type_as(inv_freqs), inv_freqs)
    return freqs.repeat_interleave(2, dim=-1).reshape(max_seqlen, 1, 1, -1), torch.cat((freqs, freqs), dim=-1).reshape(max_seqlen, 1, 1, d_model)

def get_input(
    seqlen: int,
    batch_size: int,
    num_heads: int,
    d_model: int,
):
    # set random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    cudnn.benchmark = False
    cudnn.deterministic = True

    input_interleaved = torch.randn(seqlen, batch_size, num_heads, d_model)
    input_strided = torch.cat((input_interleaved[..., ::2], input_interleaved[..., 1::2]), dim=-1)
    return input_interleaved, input_strided
    

if __name__ == "__main__":
    MAX_SEQLEN = 1024
    BATCH_SIZE = 10
    SEQLEN = 256
    NUM_HEADS = 96
    D_MODEL = 128

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    freqs_interleaved, freqs_strided = get_freqs(MAX_SEQLEN, D_MODEL)
    freqs_interleaved = freqs_interleaved.to(device)
    freqs_strided = freqs_strided.to(device)
    input_interleaved, input_strided = get_input(SEQLEN, BATCH_SIZE, NUM_HEADS, D_MODEL)
    input_interleaved = input_interleaved.to(device)
    input_strided = input_strided.to(device)

    # interleaved naive impl
    input = input_interleaved.detach().clone().requires_grad_(True)
    output_naive = rope_interleaved_torch(input, freqs_interleaved)
    output_naive.backward(gradient=torch.ones_like(output_naive))
    grad_naive = input.grad

    # interleaved triton impl
    input = input_interleaved.detach().clone().requires_grad_(True)
    output_triton = rope_interleaved_triton(input, freqs_interleaved)
    output_triton.backward(gradient=torch.ones_like(output_triton))
    grad_triton = input.grad

    torch.testing.assert_close(output_naive, output_triton)
    torch.testing.assert_close(grad_naive, grad_triton)

    # strided triton impl
    input = input_strided.detach().clone().requires_grad_(True)
    output_cuda = rope_strided_triton(input, freqs_strided)
    output_cuda.backward(gradient=torch.ones_like(output_cuda))
    grad_cuda = input.grad

    output_tr = torch.cat((output_triton[..., ::2], output_triton[..., 1::2]), dim=-1)
    grad_tr = torch.cat((grad_triton[..., ::2], grad_triton[..., 1::2]), dim=-1)
    torch.testing.assert_close(output_tr, output_cuda)
    torch.testing.assert_close(grad_tr, grad_cuda)

    # benchmark
    @triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['SEQ_LEN'],
        x_vals=[128 * i for i in range(2, 9)],
        line_arg='provider',
        line_vals=[
            'torch-interleaved',
            'triton-interleaved',
            'torch-strided',
            'triton-strided',
        ],
        line_names=[
            "Torch (interleaved)",
            "Triton (interleaved)",
            "Torch (strided)",
            "Triton (strided)",
        ],
        styles=[('blue', '-'), ('green', '-'), ('orange', ':'), ('red', '--')],
        ylabel="ms",
        plot_name="rope-performance (batch_size: 10, num_heads: 96, head_dim: 128)", 
        args={
            'BATCH_SIZE': 10,
            'NUM_HEADS': 96,
            'HEAD_DIM': 128,
        },
    ))
    def benchmark(SEQ_LEN, BATCH_SIZE, NUM_HEADS, HEAD_DIM, provider):
        x = torch.randn(SEQ_LEN, BATCH_SIZE, NUM_HEADS, HEAD_DIM, device='cuda', dtype=torch.float32)
        percentiles = [0.5, 0.2, 0.8]
        if provider == 'torch-interleaved':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope_interleaved_torch(x, freqs_interleaved), percentiles=percentiles)
        if provider == 'triton-interleaved':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope_interleaved_triton(x, freqs_interleaved), percentiles=percentiles)
        if provider == 'torch-strided':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope_strided_torch(x, freqs_interleaved), percentiles=percentiles)
        if provider == 'triton-strided':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope_strided_triton(x, freqs_strided), percentiles=percentiles)
        return ms, max_ms, min_ms

    benchmark.run(show_plots=True, print_data=True)