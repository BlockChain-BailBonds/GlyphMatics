#!/data/data/com.termux/files/usr/bin/sh
set -eu

ROOT=${GM918_ROOT:-$(CDPATH= cd -- "$(dirname "$0")" && pwd)}
LLAMA_REV=5676dfbe01e7ebe0387913680eee0ed39bf4a363
LLAMA_DIR="$ROOT/third_party/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build-gm918"
MODEL_DIR="$ROOT/models-source"
VAULT_DIR="$ROOT/vault-fast"
MODEL="$MODEL_DIR/Qwen3-1.7B-Q4_K_M.gguf"
PNG="$VAULT_DIR/918_Qwen3_1.7B.fast.png"
MODEL_SHA=d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5
MODEL_URL=https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf

pkg update -y
pkg install -y git clang cmake make curl libandroid-spawn openssl-tool

mkdir -p "$ROOT/third_party" "$MODEL_DIR" "$VAULT_DIR"
if [ ! -d "$LLAMA_DIR/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
git -C "$LLAMA_DIR" fetch --depth 1 origin "$LLAMA_REV"
git -C "$LLAMA_DIR" checkout --detach "$LLAMA_REV"
PATCH="$ROOT/patches/llama.cpp-gm918-png.patch"
if git -C "$LLAMA_DIR" apply --check "$PATCH"; then
    git -C "$LLAMA_DIR" apply "$PATCH"
elif git -C "$LLAMA_DIR" apply --reverse --check "$PATCH"; then
    echo "GM918 loader patch is already applied"
else
    echo "GM918 loader patch does not match pinned source" >&2
    exit 1
fi

cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENMP=OFF \
    -DGGML_VULKAN=OFF \
    -DLLAMA_CURL=OFF
cmake --build "$BUILD_DIR" --target llama-cli -j2

if [ ! -f "$MODEL" ]; then
    curl -fL --retry 3 "$MODEL_URL" -o "$MODEL.partial"
    printf '%s  %s\n' "$MODEL_SHA" "$MODEL.partial" | sha256sum -c -
    mv "$MODEL.partial" "$MODEL"
fi

if [ ! -f "$PNG" ]; then
    "$ROOT/sigilvm4/encode_mmap_png.py" "$MODEL" "$PNG.partial"
    "$ROOT/sigilvm4/verify_mmap_png.py" "$PNG.partial" \
        --require-sprites --sha256 "$MODEL_SHA"
    mv "$PNG.partial" "$PNG"
fi

"$ROOT/sigilvm4/verify_mmap_png.py" "$PNG" \
    --require-sprites --sha256 "$MODEL_SHA"
chmod +x "$ROOT/bin/offline-codex"
ln -sf "$ROOT/bin/offline-codex" "$PREFIX/bin/offline-codex"

echo "Installed. Run: offline-codex"
echo "The verified source model was preserved at: $MODEL"
