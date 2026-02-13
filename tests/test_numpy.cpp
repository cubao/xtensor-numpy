#include <pybind11/embed.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

namespace py = pybind11;

int main(int argc, char **argv) {
    if (argc < 1) {
        std::cerr << "Usage:\n\t" << argv[0] << " <path/to/script.py>" << std::endl;
        return -1;
    }
    try {
        // Initialize the Python interpreter
        py::scoped_interpreter guard{};

        // Path to the script
        const std::string script_path = argv[1];

        // Open the script file
        std::ifstream file(script_path);
        if (!file.is_open()) {
            std::cerr << "Could not open " << script_path << " script file." << std::endl;
            return 1;
        }

        // Read and execute the script into a string
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string script = buffer.str();
        py::exec(script);
        std::cout << "Numpy script executed successfully." << std::endl;
    }
    catch (const py::python_error& e) {
        // Catch and print Python exceptions
        std::cerr << "Python error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        // Catch and print other C++ exceptions
        std::cerr << "C++ exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
