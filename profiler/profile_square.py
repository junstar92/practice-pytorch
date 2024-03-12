import torch

# using CUDA Event
def time_pytorch_function(func, input):
    # CUDA is async, so we can't use python time module
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # warmup
    for _ in range(5):
        func(input)
    
    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end)

# using PyTorch profiler
b = torch.randn(10000, 10000).cuda()

def square_2(a):
    return a * a

def square_3(a):
    return a ** 2

print("torch.square: ", time_pytorch_function(torch.square, b), " ms")
print("square_2: ", time_pytorch_function(square_2, b), " ms")
print("square_3: ", time_pytorch_function(square_3, b), " ms")
print("square_4 (triton): ", time_pytorch_function(square_4, b), " ms")

print("=============")
print("Profiling torch.square")
print("=============")

# Now profile each function using pytorch profiler
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    torch.square(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a * a")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_2(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a ** 2")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_3(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))