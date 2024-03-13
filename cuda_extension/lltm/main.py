import torch
import time

# first, run 'python setup.py install(or develop)'
import lltm_torch
import lltm_cpp_impl
import lltm_cuda_impl

if __name__ == "__main__":
    batch_size = 16
    input_features = 32
    state_size = 128

    X = torch.randn(batch_size, input_features)
    h = torch.randn(batch_size, state_size)
    C = torch.randn(batch_size, state_size)

    # torch version
    rnn_torch = lltm_torch.LLTM(input_features, state_size)

    forward = 0
    backward = 0
    for _ in range(10000):
        start = time.time()
        new_h, new_C = rnn_torch(X, (h, C))
        forward += time.time() - start

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        backward += time.time() - start
    
    print(f"[Torch] Forward: {forward * 1e6/1e5:.3f} us | Backward {backward * 1e6/1e5:.3f} us")

    # cpp version
    rnn_cpp = lltm_cpp_impl.LLTM(input_features, state_size)

    forward = 0
    backward = 0
    for _ in range(10000):
        start = time.time()
        new_h, new_C = rnn_cpp(X, (h, C))
        forward += time.time() - start

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        backward += time.time() - start
    
    print(f"[CPP] Forward: {forward * 1e6/1e5:.3f} us | Backward {backward * 1e6/1e5:.3f} us")

    # cuda version
    device = 'cuda'
    rnn_cuda = lltm_cuda_impl.LLTM(input_features, state_size).to(device)
    X = X.to(device)
    h = h.to(device)
    C = C.to(device)

    forward = 0
    backward = 0
    for _ in range(10000):
        start = time.time()
        new_h, new_C = rnn_cuda(X, (h, C))
        forward += time.time() - start

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        backward += time.time() - start
    
    print(f"[CUDA] Forward: {forward * 1e6/1e5:.3f} us | Backward {backward * 1e6/1e5:.3f} us")

