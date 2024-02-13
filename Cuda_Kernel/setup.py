from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='FastAttention',
    ext_modules=[
        CUDAExtension('FastAttention', [
            # 'custom_op_kernel.cu',
            # 'custom_op.cpp',
            'cu_code_fast.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
