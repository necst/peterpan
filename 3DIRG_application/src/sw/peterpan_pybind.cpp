#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "peterpan_runner.hpp"

namespace py = pybind11;

PYBIND11_MODULE(peterpan_cpp, m) {
    m.doc() = "Python bindings for PeterPan runner";

    m.def(
        "run",
        [](const std::vector<std::string>& args) {
            int ret = peterpan_runner(args);
            if (ret != 0) {
                throw std::runtime_error("peterpan_runner failed with exit code " + std::to_string(ret));
            }
        },
        py::arg("args"),
        R"pbdoc(
Run PeterPan with the same arguments normally passed on the command line.

Example:
    import peterpan_cpp
    peterpan_cpp.run([
        "fixed.nii",
        "moving.nii",
        "64",
        "--type=nii",
        "--num_levels=4"
    ])
)pbdoc");
}