from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='FastAttention2',
    ext_modules=[
        # CUDAExtension('FastAttention.forward', [
        #     'forwardv2.cu',
        # ]),
        # CUDAExtension('FastAttention.backward', [
        #     'backwardv2_comb.cu',
        # ]),
        CUDAExtension('FastAttention2.kernel', [
            'combined_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
