// hm_host: keeps a GGUF model loaded and, for each request on stdin, returns per-token
// post-norm hidden states (h_nextn = the vector the LM head reads) and the MoE router's
// expert choices for every layer. Used by the hippocampal-memory experiments (hm.py).
//
// usage: hm_host <model.gguf> [n_cpu_moe_all=1] [ctx=2048]
// stdin  request: int32 type; type 1 = tokenize: int32 nbytes, bytes  -> int32 n, int32 tok[n]
//                             type 2 = extract : int32 n, int32 tok[n] -> int32 n, int32 d, float h[n*d],
//                                                                         int32 L, int32 k, int32 ids[L*n*k]
//                             type 3 = set LoRA: int32 nbytes, path bytes (0 = clear), float scale -> int32 ok
//                             type 0 = quit
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "llama.h"
#include "ggml-backend.h"

static FARPROC findsym(const char * n) {
    const char * libs[] = { "llama.dll", "ggml.dll", "ggml-base.dll" };
    for (auto l : libs) { HMODULE h = GetModuleHandleA(l); if (h) { FARPROC p = GetProcAddress(h, n); if (p) return p; } }
    return nullptr;
}
#define LOADFN(name) auto p_##name = (decltype(&name))findsym(#name); \
    if (!p_##name) { fprintf(stderr, "missing symbol %s\n", #name); return 1; }

struct st_t {
    decltype(&ggml_backend_tensor_get) tget = nullptr;
    int n_layer = 0, k = 0, d = 0;
    std::vector<float> h;            // [n*d] for the current request
    std::vector<int32_t> ids;        // [L*n*k]
    int n = 0, base = 0;             // tokens in request, offset of current ubatch
    std::vector<std::string> cap_names;        // extra tensors to capture (env HM_CAPTURE, comma separated)
    std::vector<int> cap_width;                // floats per token
    std::vector<std::vector<float>> cap;       // [n*width]
};

static int cap_index(st_t * st, const char * name) {
    for (size_t i = 0; i < st->cap_names.size(); ++i) if (st->cap_names[i] == name) return (int) i;
    return -1;
}

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = (st_t *) ud;
    const bool is_h = strcmp(t->name, "h_nextn") == 0;
    const bool is_topk = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    const int ci = cap_index(st, t->name);
    if (ask) return is_h || is_topk || ci >= 0;
    if (ci >= 0 && t->type == GGML_TYPE_F32) {
        // token axis is the last non-trivial axis: ne[2] for [a,b,n] tensors, ne[1] for [a,n]
        const bool three = strncmp(t->name, "ffn_moe_swiglu", 14) == 0 || t->ne[2] > 1;  // [ff, k, n] even when n == 1
        const int64_t ntok = three ? t->ne[2] : t->ne[1];
        const int64_t w = three ? t->ne[0] * t->ne[1] : t->ne[0];
        st->cap_width[ci] = (int) w;
        auto & buf = st->cap[ci];
        if ((int64_t) buf.size() < (int64_t) st->n * w) buf.resize((size_t) st->n * w);
        for (int64_t j = 0; j < ntok && st->base + j < st->n; ++j) {
            float * dst = buf.data() + (st->base + j) * w;
            if (three) {
                for (int64_t r = 0; r < t->ne[1]; ++r)
                    st->tget(t, dst + r * t->ne[0], (size_t)(j * t->nb[2] + r * t->nb[1]), (size_t)(t->ne[0] * sizeof(float)));
            } else {
                st->tget(t, dst, (size_t)(j * t->nb[1]), (size_t)(w * sizeof(float)));
            }
        }
        return true;
    }
    const int64_t rows = t->ne[1];
    if (is_h) {
        const int64_t d = t->ne[0];
        st->d = (int) d;
        if ((int64_t) st->h.size() < (int64_t)st->n * d) st->h.resize((size_t)st->n * d);
        for (int64_t j = 0; j < rows && st->base + j < st->n; ++j)
            st->tget(t, st->h.data() + (st->base + j) * d, (size_t)(j * t->nb[1]), (size_t)(d * sizeof(float)));
        return true;
    }
    if (is_topk && t->type == GGML_TYPE_I32) {
        int il = atoi(t->name + 13);
        int64_t k = t->ne[0];
        st->k = (int) k;
        size_t need = (size_t)st->n_layer * st->n * k;
        if (st->ids.size() < need) st->ids.assign(need, -1);
        for (int64_t j = 0; j < rows && st->base + j < st->n; ++j)
            st->tget(t, st->ids.data() + ((size_t)il * st->n + st->base + j) * k, (size_t)(j * t->nb[1]), (size_t)(k * sizeof(int32_t)));
    }
    return true;
}

static bool rd(void * p, size_t n) { return fread(p, 1, n, stdin) == n; }
static void wr(const void * p, size_t n) { fwrite(p, 1, n, stdout); }

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: hm_host model [cpu_moe] [ctx]\n"); return 1; }
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    int cpu_moe = argc > 2 ? atoi(argv[2]) : 1;
    int n_ctx = argc > 3 ? atoi(argv[3]) : 2048;
    if (!LoadLibraryA("ggml.dll") || !LoadLibraryA("ggml-base.dll") || !LoadLibraryA("llama.dll")) { fprintf(stderr, "dll load failed\n"); return 1; }
    LOADFN(ggml_backend_load_all); LOADFN(ggml_backend_tensor_get); LOADFN(ggml_backend_dev_by_type); LOADFN(ggml_backend_dev_buffer_type);
    LOADFN(llama_backend_init); LOADFN(llama_model_default_params); LOADFN(llama_context_default_params);
    LOADFN(llama_model_load_from_file); LOADFN(llama_init_from_model); LOADFN(llama_model_get_vocab); LOADFN(llama_tokenize);
    LOADFN(llama_batch_get_one); LOADFN(llama_decode); LOADFN(llama_get_memory); LOADFN(llama_memory_clear);
    LOADFN(llama_model_n_layer); LOADFN(llama_set_embeddings_nextn);
    LOADFN(llama_adapter_lora_init); LOADFN(llama_set_adapters_lora); LOADFN(llama_adapter_lora_free);
    p_ggml_backend_load_all(); p_llama_backend_init();

    st_t st; st.tget = p_ggml_backend_tensor_get;
    if (const char * c = getenv("HM_CAPTURE")) {
        std::string s(c), cur;
        for (char ch : s + ",") { if (ch == ',') { if (!cur.empty()) st.cap_names.push_back(cur); cur.clear(); } else cur += ch; }
        st.cap_width.assign(st.cap_names.size(), 0);
        st.cap.assign(st.cap_names.size(), {});
    }
    llama_model_params mp = p_llama_model_default_params();
    mp.n_gpu_layers = 99;
    static llama_model_tensor_buft_override ov[2];
    if (cpu_moe) {
        const char * pat = getenv("HM_CPU_PATTERN");  // e.g. blk\.(1[4-9]|[23][0-9])\.ffn_.*_exps = --n-cpu-moe 26
        ov[0].pattern = pat ? pat : "\\.ffn_(up|down|gate|gate_up)_exps";
        ov[0].buft = p_ggml_backend_dev_buffer_type(p_ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
        ov[1].pattern = nullptr; ov[1].buft = nullptr;
        mp.tensor_buft_overrides = ov;
    }
    llama_model * model = p_llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    st.n_layer = p_llama_model_n_layer(model);
    const llama_vocab * vocab = p_llama_model_get_vocab(model);
    llama_context_params cp = p_llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.cb_eval = cb_eval; cp.cb_eval_user_data = &st;
    llama_context * ctx = p_llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    p_llama_set_embeddings_nextn(ctx, true, false);  // keep h for every token, not only the last
    fprintf(stderr, "hm_host ready: layers=%d\n", st.n_layer);
    fflush(stderr);

    llama_adapter_lora * cur_lora = nullptr;
    for (;;) {
        int32_t type;
        if (!rd(&type, 4) || type == 0) break;
        if (type == 1) {
            int32_t nb; rd(&nb, 4);
            std::string s((size_t)nb, '\0'); rd(s.data(), (size_t)nb);
            std::vector<llama_token> toks((size_t)nb + 16);
            int n = p_llama_tokenize(vocab, s.c_str(), nb, toks.data(), (int32_t)toks.size(), false, true);
            if (n < 0) n = 0;
            wr(&n, 4); wr(toks.data(), (size_t)n * 4);
        } else if (type == 2) {
            int32_t n; rd(&n, 4);
            std::vector<llama_token> toks((size_t)n); rd(toks.data(), (size_t)n * 4);
            st.n = n; st.h.assign(0, 0.f); st.ids.assign(0, -1);
            for (auto & b : st.cap) b.assign(0, 0.f);
            p_llama_memory_clear(p_llama_get_memory(ctx), true);
            int ok = 1;
            for (int i = 0; i < n; i += 512) {
                int m = (n - i) < 512 ? (n - i) : 512;
                st.base = i;
                llama_batch b = p_llama_batch_get_one(toks.data() + i, m);
                if (p_llama_decode(ctx, b) != 0) { ok = 0; break; }
            }
            int32_t nn = ok ? n : 0, d = st.d, L = st.n_layer, k = st.k;
            if ((int)st.h.size() < nn * d) st.h.resize((size_t)nn * d, 0.f);
            if ((int)st.ids.size() < L * nn * k) st.ids.resize((size_t)L * nn * k, -1);
            wr(&nn, 4); wr(&d, 4); wr(st.h.data(), (size_t)nn * d * 4);
            wr(&L, 4); wr(&k, 4); wr(st.ids.data(), (size_t)L * nn * k * 4);
            int32_t nc = (int32_t) st.cap.size();
            wr(&nc, 4);
            for (int32_t c = 0; c < nc; ++c) {
                int32_t w = st.cap_width[c];
                if ((int)st.cap[c].size() < nn * w) st.cap[c].resize((size_t)nn * w, 0.f);
                wr(&w, 4); wr(st.cap[c].data(), (size_t)nn * w * 4);
            }
        }
        else if (type == 3) {
            int32_t nb; rd(&nb, 4);
            std::string path((size_t)nb, '\0'); if (nb) rd(path.data(), (size_t)nb);
            float scale; rd(&scale, 4);
            p_llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
            if (cur_lora) { p_llama_adapter_lora_free(cur_lora); cur_lora = nullptr; }
            int32_t ok = 1;
            if (nb) {
                cur_lora = p_llama_adapter_lora_init(model, path.c_str());
                if (!cur_lora) ok = 0;
                else ok = p_llama_set_adapters_lora(ctx, &cur_lora, 1, &scale) == 0;
            }
            fprintf(stderr, "lora '%s' scale %.3f -> %d\n", path.c_str(), scale, ok); fflush(stderr);
            wr(&ok, 4);
        }
        fflush(stdout);
    }
    return 0;
}
