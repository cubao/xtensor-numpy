#include <stdio.h>
#include <stdlib.h>
#include "pocketpy/pocketpy.h"

extern "C" bool py_module_initialize();

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage:\n\t%s <path/to/script.py>\n", argv[0]);
        return -1;
    }
    try {
        py_initialize();
        py_sys_setargv(argc, argv);

        // Register the numpy module (linked into this binary)
        py_module_initialize();

        // Read the script file
        const char* filename = argv[1];
        FILE* f = fopen(filename, "rb");
        if (!f) {
            fprintf(stderr, "Could not open '%s'\n", filename);
            py_finalize();
            return 1;
        }
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        char* data = (char*)malloc(size + 1);
        size = fread(data, 1, size, f);
        data[size] = 0;
        fclose(f);

        // Execute the script
        bool ok = py_exec(data, filename, EXEC_MODE, NULL);
        if (!ok) {
            py_printexc();
        }

        free(data);
        int code = ok ? 0 : 1;
        py_finalize();
        return code;
    catch (const std::exception& e) {
        // Catch and print other C++ exceptions
        std::cerr << "C++ exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
