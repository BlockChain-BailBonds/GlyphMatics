#!/data/data/com.termux/files/usr/bin/sh
set -eu
clang++ -std=c++17 -O3 -DNDEBUG gm918_png_runtime.cpp -lz -lcrypto -o gm918-png-runtime
strip gm918-png-runtime
./gm918-png-runtime 2>/dev/null || test "$?" -eq 2
