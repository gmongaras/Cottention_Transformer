from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_op',
    ext_modules=[
        CUDAExtension('custom_op', [
            # 'custom_op_kernel.cu',
            # 'custom_op.cpp',
            'cu_code_fast.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
