#include <pybind11/pybind11.h>
extern "C" {
    #include "run.c"  // 直接包含 C 源代碼
}

namespace py = pybind11;

PYBIND11_MODULE(llama_model, m) {
    m.doc() = "pybind11 plugin for Llama-2 Transformer model";

    py::class_<Transformer>(m, "Transformer")
        .def(py::init<>())
        .def("generate", &Transformer::generate);
    
    // 您可以按需添加更多類別或函數的綁定
}