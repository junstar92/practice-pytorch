from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

# python setup.py install : for installing package
# python setup.py develop : for installing editable package
if __name__ == "__main__":
    setup(
        name='lltm_cpp',
        ext_modules=[
            CppExtension(
                name='lltm_cpp',
                sources=['lltm_cpp.cpp'],
            ),
            CUDAExtension(
                name='lltm_cuda',
                sources=[
                    'lltm_cuda.cpp',
                    'lltm_kernel.cu',
                ],
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        },
    )