#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "run.h"
namespace py = pybind11;

PYBIND11_MODULE(llama_model, m) {
    m.doc() = "pybind11 plugin for Llama-2 c code";

    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("dim", &Config::dim)
        .def_readwrite("hidden_dim", &Config::hidden_dim)
        .def_readwrite("n_layers", &Config::n_layers)
        .def_readwrite("n_heads", &Config::n_heads)
        .def_readwrite("n_kv_heads", &Config::n_kv_heads)
        .def_readwrite("vocab_size", &Config::vocab_size)
        .def_readwrite("seq_len", &Config::seq_len);

    py::class_<TransformerWeights>(m, "TransformerWeights")
        .def(py::init<>())  
        .def_readwrite("token_embedding_table", &TransformerWeights::token_embedding_table)
        .def_readwrite("rms_att_weight", &TransformerWeights::rms_att_weight)
        .def_readwrite("rms_ffn_weight", &TransformerWeights::rms_ffn_weight)
        .def_readwrite("wq", &TransformerWeights::wq)
        .def_readwrite("wk", &TransformerWeights::wk)
        .def_readwrite("wv", &TransformerWeights::wv)
        .def_readwrite("wo", &TransformerWeights::wo)
        .def_readwrite("w1", &TransformerWeights::w1)
        .def_readwrite("w2", &TransformerWeights::w2)
        .def_readwrite("w3", &TransformerWeights::w3)
        .def_readwrite("rms_final_weight", &TransformerWeights::rms_final_weight)
        .def_readwrite("wcls", &TransformerWeights::wcls);

    py::class_<RunState>(m, "RunState")
        .def(py::init<>())
        .def_readwrite("x", &RunState::x)
        .def_readwrite("xb", &RunState::xb)
        .def_readwrite("xb2", &RunState::xb2)
        .def_readwrite("hb", &RunState::hb)
        .def_readwrite("hb2", &RunState::hb2)
        .def_readwrite("q", &RunState::q)
        .def_readwrite("k", &RunState::k)
        .def_readwrite("v", &RunState::v)
        .def_readwrite("att", &RunState::att)
        .def_readwrite("logits", &RunState::logits)
        .def_readwrite("key_cache", &RunState::key_cache)
        .def_readwrite("value_cache", &RunState::value_cache);



    py::class_<Transformer>(m, "Transformer")
        .def(py::init<>())
        .def_readwrite("config", &Transformer::config)
        .def_readwrite("weights", &Transformer::weights)
        .def_readwrite("state", &Transformer::state)
        .def_readwrite("fd", &Transformer::fd)
        .def_readwrite("data", &Transformer::data)
        .def_readwrite("file_size", &Transformer::file_size);



    // // m.def("error_usage", &error_usage);
    m.def("build_transformer", &build_transformer, "Builds the transformer");
    m.def("build_tokenizer", &build_tokenizer);
    m.def("build_sampler", &build_sampler);
    m.def("generate", &generate);
    m.def("chat", &chat);

    
    
    m.def("goLLM", &goLLM, "Builds the transformer");
}