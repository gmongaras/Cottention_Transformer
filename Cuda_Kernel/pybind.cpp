#include <torch/extension.h>

#include "forward.cu"
#include "backward.cu"



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Forward definitionz
    m.def("forward_float32", &forward_<float>);
    m.def("forward_float16", &forward_<at::Half>);
    try {
        m.def("forward_bfloat16", &forward_<at::BFloat16>);
    } catch (const std::exception& e) {
        std::cout << "GPU does not support bfloat16. Skipping..." << std::endl;
        // std::cerr << "Error: " << e.what() << std::endl;
    }

    // Backward definitions
    m.def("backward_float32", &backward_<float>);
    m.def("backward_float16", &backward_<at::Half>);
    try {
        m.def("backward_bfloat16", &backward_<at::BFloat16>);
    } catch (const std::exception& e) {
        std::cout << "GPU does not support bfloat16. Skipping..." << std::endl;
        // std::cerr << "Error: " << e.what() << std::endl;
    }
}