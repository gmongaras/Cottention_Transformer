from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='FastAttention',
    ext_modules=[
        CUDAExtension('FastAttention.forward', [
            'forward.cu',
        ]),
        CUDAExtension('FastAttention.backward', [
            'backward.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
