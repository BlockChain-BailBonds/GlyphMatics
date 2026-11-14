#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <openssl/sha.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <zlib.h>

namespace {
constexpr unsigned char PNG_SIG[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
constexpr char MAGIC[8] = {'G','M','9','1','8','P','4','\0'};
constexpr uint32_t VERSION = 4;
constexpr uint32_t KIND_MODEL = 1;

int create_memfd(const char *name) {
#ifdef __NR_memfd_create
    return int(syscall(__NR_memfd_create, name, 0));
#else
    errno = ENOSYS;
    return -1;
#endif
}

uint32_t be32(const unsigned char *p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
}
uint64_t be64(const unsigned char *p) { return (uint64_t(be32(p)) << 32) | be32(p + 4); }

std::vector<unsigned char> read_file(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("open failed: " + path + ": " + strerror(errno));
    if (fseek(f, 0, SEEK_END) || ftell(f) < 0) { fclose(f); throw std::runtime_error("seek failed: " + path); }
    long size = ftell(f); rewind(f);
    std::vector<unsigned char> out(size);
    if (size && fread(out.data(), 1, size, f) != size_t(size)) { fclose(f); throw std::runtime_error("read failed: " + path); }
    fclose(f); return out;
}

struct Header {
    uint32_t version{}, kind{}, index{};
    uint64_t offset{};
    uint32_t length{}, total{};
    std::array<unsigned char, 32> sha{};
};
struct Decoded { Header h; std::vector<unsigned char> payload; };

std::string hex_digest(const unsigned char *data, size_t len) {
    static const char digits[] = "0123456789abcdef";
    std::string out; out.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) { out.push_back(digits[data[i] >> 4]); out.push_back(digits[data[i] & 15]); }
    return out;
}

std::string json_string(const std::string &json, const std::string &key) {
    std::string needle = "\"" + key + "\":\""; size_t start = json.find(needle);
    if (start == std::string::npos) throw std::runtime_error("manifest missing " + key);
    start += needle.size(); size_t end = json.find('"', start);
    if (end == std::string::npos) throw std::runtime_error("invalid manifest string " + key);
    return json.substr(start, end - start);
}

uint64_t json_uint(const std::string &json, const std::string &key) {
    std::string needle = "\"" + key + "\":"; size_t start = json.find(needle);
    if (start == std::string::npos) throw std::runtime_error("manifest missing " + key);
    start += needle.size(); char *end = nullptr; unsigned long long value = strtoull(json.c_str() + start, &end, 10);
    if (end == json.c_str() + start) throw std::runtime_error("invalid manifest integer " + key);
    return value;
}

Decoded decode_png(const std::string &path) {
    auto file = read_file(path);
    if (file.size() < 8 || memcmp(file.data(), PNG_SIG, 8)) throw std::runtime_error("invalid PNG signature: " + path);
    size_t pos = 8; uint32_t width = 0, height = 0; Header h; bool have_header = false, have_ihdr = false, have_iend = false;
    std::vector<unsigned char> idat;
    while (pos + 12 <= file.size()) {
        uint32_t len = be32(file.data() + pos); pos += 4;
        if (pos + 4ULL + len + 4ULL > file.size()) throw std::runtime_error("truncated PNG chunk: " + path);
        const unsigned char *type = file.data() + pos; pos += 4;
        const unsigned char *data = file.data() + pos; pos += len;
        uint32_t recorded_crc = be32(file.data() + pos); pos += 4;
        uLong crc = crc32(0, type, 4); crc = crc32(crc, data, len);
        if (uint32_t(crc) != recorded_crc) throw std::runtime_error("PNG CRC mismatch: " + path);
        if (!memcmp(type, "IHDR", 4)) {
            if (len != 13 || data[8] != 8 || data[9] != 0 || data[12] != 0) throw std::runtime_error("unsupported PNG layout: " + path);
            width = be32(data); height = be32(data + 4); have_ihdr = true;
        } else if (!memcmp(type, "gmHD", 4)) {
            if (len != 68 || memcmp(data, MAGIC, 8)) throw std::runtime_error("invalid GM918 header: " + path);
            h.version = be32(data + 8); h.kind = be32(data + 12); h.index = be32(data + 16);
            h.offset = be64(data + 20); h.length = be32(data + 28); h.total = be32(data + 32);
            memcpy(h.sha.data(), data + 36, 32); have_header = true;
        } else if (!memcmp(type, "IDAT", 4)) idat.insert(idat.end(), data, data + len);
        else if (!memcmp(type, "IEND", 4)) { have_iend = true; break; }
    }
    if (!have_ihdr || !have_header || !have_iend || h.version != VERSION) throw std::runtime_error("incomplete/unsupported GM918 PNG: " + path);
    size_t scan_size = size_t(height) * (size_t(width) + 1); std::vector<unsigned char> scan(scan_size); uLongf out_len = scan_size;
    int zr = uncompress(scan.data(), &out_len, idat.data(), idat.size());
    if (zr != Z_OK || out_len != scan_size) throw std::runtime_error("PNG inflate failed: " + path);
    Decoded out; out.h = h; out.payload.reserve(size_t(width) * height);
    for (uint32_t y = 0; y < height; ++y) {
        size_t row = size_t(y) * (width + 1);
        if (scan[row] != 0) throw std::runtime_error("unsupported PNG filter: " + path);
        out.payload.insert(out.payload.end(), scan.begin() + row + 1, scan.begin() + row + 1 + width);
    }
    if (out.payload.size() < h.length) throw std::runtime_error("short GM918 payload: " + path);
    out.payload.resize(h.length);
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256(out.payload.data(), out.payload.size(), digest.data());
    if (!std::equal(digest.begin(), digest.end(), h.sha.begin())) throw std::runtime_error("GM918 payload SHA-256 mismatch: " + path);
    return out;
}

std::vector<std::string> tile_paths(const std::string &vault) {
    DIR *d = opendir(vault.c_str()); if (!d) throw std::runtime_error("cannot open vault: " + vault);
    std::vector<std::string> paths;
    while (dirent *e = readdir(d)) {
        std::string n = e->d_name;
        if (n.rfind("tile-", 0) == 0 && n.size() > 4 && n.substr(n.size() - 4) == ".png") paths.push_back(vault + "/" + n);
    }
    closedir(d); std::sort(paths.begin(), paths.end()); return paths;
}

void write_all_at(int fd, const unsigned char *data, size_t len, off_t offset) {
    size_t done = 0;
    while (done < len) {
        ssize_t n = pwrite(fd, data + done, len - done, offset + done);
        if (n < 0) { if (errno == EINTR) continue; throw std::runtime_error("memfd write failed"); }
        done += size_t(n);
    }
}

int load_memfd(const std::string &vault, bool verify_only) {
    auto paths = tile_paths(vault); if (paths.empty()) throw std::runtime_error("vault has no model PNG tiles");
    auto manifest_tile = decode_png(vault + "/manifest.png");
    if (manifest_tile.h.kind != 2) throw std::runtime_error("manifest.png has wrong GM918 kind");
    std::string manifest(manifest_tile.payload.begin(), manifest_tile.payload.end());
    if (json_uint(manifest, "version") != VERSION) throw std::runtime_error("unsupported manifest version");
    if (json_uint(manifest, "tile_count") != paths.size()) throw std::runtime_error("manifest tile count mismatch");
    uint64_t manifest_bytes = json_uint(manifest, "model_bytes");
    std::string manifest_sha = json_string(manifest, "model_sha256");
    int fd = verify_only ? -1 : create_memfd("gm918-model");
    if (!verify_only && fd < 0) throw std::runtime_error("memfd_create failed");
    uint64_t expected_offset = 0; uint32_t expected_total = paths.size(); SHA256_CTX whole; SHA256_Init(&whole);
    for (size_t i = 0; i < paths.size(); ++i) {
        auto tile = decode_png(paths[i]);
        if (tile.h.kind != KIND_MODEL || tile.h.index != i || tile.h.total != expected_total || tile.h.offset != expected_offset)
            throw std::runtime_error("tile index/offset mismatch: " + paths[i]);
        if (!verify_only) write_all_at(fd, tile.payload.data(), tile.payload.size(), tile.h.offset);
        SHA256_Update(&whole, tile.payload.data(), tile.payload.size());
        expected_offset += tile.payload.size();
    }
    std::array<unsigned char, SHA256_DIGEST_LENGTH> whole_digest{}; SHA256_Final(whole_digest.data(), &whole);
    if (expected_offset != manifest_bytes) throw std::runtime_error("manifest model size mismatch");
    if (hex_digest(whole_digest.data(), whole_digest.size()) != manifest_sha) throw std::runtime_error("manifest model SHA-256 mismatch");
    std::cerr << "GM918_PNG_VERIFIED tiles=" << paths.size() << " bytes=" << expected_offset << "\n";
    if (!verify_only && ftruncate(fd, expected_offset)) throw std::runtime_error("memfd truncate failed");
    return fd;
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 3 || (std::string(argv[1]) != "verify" && std::string(argv[1]) != "run")) {
            std::cerr << "usage: gm918-png-runtime verify VAULT | run VAULT [llama-cli arguments...]\n"; return 2;
        }
        bool verify = std::string(argv[1]) == "verify";
        int fd = load_memfd(argv[2], verify);
        if (verify) return 0;
        std::string model = "/proc/self/fd/" + std::to_string(fd);
        std::vector<char *> args; args.push_back(const_cast<char *>("llama-cli"));
        args.push_back(const_cast<char *>("-m")); args.push_back(model.data());
        for (int i = 3; i < argc; ++i) args.push_back(argv[i]);
        args.push_back(nullptr);
        execvp("llama-cli", args.data());
        throw std::runtime_error(std::string("exec llama-cli failed: ") + strerror(errno));
    } catch (const std::exception &e) {
        std::cerr << "GM918_ERROR " << e.what() << "\n"; return 1;
    }
}
