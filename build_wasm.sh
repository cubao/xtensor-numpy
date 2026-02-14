set -e

# Run prebuild if _generated.c doesn't exist
if [ ! -f pocketpy/src/common/_generated.c ]; then
    python3 pocketpy/prebuild.py
fi

# Create output directory
rm -rf web/lib
mkdir -p web/lib

# Generate test_numpy.js from test source
python3 -c "
import json
with open('tests/test_numpy.py') as f:
    src = f.read()
print('var TEST_SOURCE = ' + json.dumps(src) + ';')
" > web/test_numpy.js

# Common flags
DEFINES="-DPK_ENABLE_OS=0 -DPK_ENABLE_THREADS=0 -DPK_ENABLE_DETERMINISM=0 \
    -DPK_ENABLE_WATCHDOG=0 -DPK_ENABLE_CUSTOM_SNAME=0 -DPK_ENABLE_MIMALLOC=0 \
    -DPK_BUILD_MODULE_LZ4 -DPK_BUILD_MODULE_MSGPACK -DNDEBUG"
INCLUDES="-Ipocketpy/include -Iinclude -Ipocketpy/3rd/lz4 -Ipocketpy/3rd/lz4/lz4/lib -Ipocketpy/3rd/msgpack/include"
WARNINGS="-Wno-sign-compare -Wno-conversion -Wno-unused-variable -Wno-unused-parameter"

# Generate a unity C file that includes all pocketpy sources
TMPDIR=$(mktemp -d)
UNITY_C="$TMPDIR/unity_pocketpy.c"

for f in $(find pocketpy/src/ -name "*.c" | sort); do
    echo "#include \"$(pwd)/$f\"" >> "$UNITY_C"
done
echo "#include \"$(pwd)/pocketpy/3rd/lz4/lz4/lib/lz4.c\"" >> "$UNITY_C"
echo "#include \"$(pwd)/pocketpy/3rd/msgpack/src/mpack.c\"" >> "$UNITY_C"
echo "#include \"$(pwd)/pocketpy/3rd/msgpack/src/bindings.c\"" >> "$UNITY_C"

# Compile unity C source with emcc
emcc "$UNITY_C" -c -Os $DEFINES $INCLUDES $WARNINGS -o "$TMPDIR/pocketpy.o"

# Compile C++ source and link with em++
em++ "$TMPDIR/pocketpy.o" \
    src/numpy.cpp \
    $INCLUDES \
    -std=c++17 -Os \
    $DEFINES \
    -DSUPPRESS_XTENSOR_WARNINGS \
    -DPY_DYNAMIC_MODULE \
    -sEXPORTED_FUNCTIONS=_py_initialize,_py_exec,_py_finalize,_py_printexc,_py_clearexc,_py_module_initialize \
    -sEXPORTED_RUNTIME_METHODS=ccall \
    -sALLOW_MEMORY_GROWTH=1 \
    -sSTACK_SIZE=1048576 \
    $WARNINGS \
    -o web/lib/pocketpy.js

rm -rf "$TMPDIR"

echo "Build complete: web/lib/pocketpy.js, web/lib/pocketpy.wasm"
