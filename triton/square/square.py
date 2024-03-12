import triton
import triton.language as tl
import torch

@triton.jit#(interpret=True)
def square_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    # the stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # the block size is the next power of two greater than n_cols, so we can fit each
    # row in a single bloxck
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # load the row into SRAM, using a mark since BLOCK_SIZE may be greater than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    square_output = row * row

    # write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)

def square(
    x: torch.Tensor
):
    n_rows, n_cols = x.shape
    # the block size is the smallest power of two greater than the number of columns in 'x'
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    
    # allocate output
    y = torch.empty_like(x)

    # enqueue kernel with 1D launch grid (one kernel instance per row of the input matrix)
    square_kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y

x = torch.randn(1823, 781, device='cuda')
y_triton = square(x)
y_torch = torch.square(x)

torch.testing.assert_close(y_triton, y_torch)

class SquareCallable:
    def __init__(self):
        super().__init__()
        self.square = torch.square
    
    def __call__(self, x):
        return self.square(x)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 11)],
        line_arg='provider',
        line_vals=[
            'triton',
            'torch-native',
            'torch-compile',
        ],
        line_names=[
            'triton',
            'torch (native)',
            'torch (compiled)',
        ],
        styles=[
            ('blue', '-'),
            ('green', '-'),
            ('green', '--'),
        ],
        ylabel="GB/s",
        plot_name="square() performance",
        args={"M": 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    percentiles = [0.5, 0.2, 0.8]
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.square(x), percentiles=percentiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: square(x), percentiles=percentiles)
    if provider == 'torch-compile':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.compile(lambda x: torch.square(x))(x), percentiles=percentiles)
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(show_plots=False, print_data=True)