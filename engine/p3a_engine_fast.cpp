// p3a_engine.cpp — the existing Hyperion C++ pattern applied to the 42M
// alpha model (HYP2 type-0 FP32): OpenMP rows, KV cache, greedy decode.
// Modes: --mode dense|addr|oracle  (addr = logreg top32 addressed FFN,
// mirrors the PyTorch port exactly: no gate/up/down bias in addr mode).
//   bpt  -> evaluate 16 val windows (matches alpha_p3a.evaluate, k=None path
//           for dense / k=32 for oracle; addr uses the trained addresser)
//   gen  -> greedy decode N tokens from prompt, print tok/s + tokens
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <omp.h>
using namespace std;
using Clock = chrono::steady_clock;
static double T_QKV = 0, T_ATTN = 0, T_PROJ = 0, T_FFN = 0, T_LOGIT = 0;
static inline double tsec() { return chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

static const int L = 8, D = 512, NH = 8, HD = 64, FFN = 2048, SEQ = 256, KTOP = 32;
int V = 16000;
int CBUDGET = 2048;

struct T { vector<uint32_t> shape; vector<float> data; };
map<string, T*> M;

static inline float* get(const string& n) {
    auto it = M.find(n);
    if (it == M.end()) { fprintf(stderr, "missing %s\n", n.c_str()); exit(1); }
    return it->second->data.data();
}

static void ln(const float* x, const float* w, const float* b, float* o) {
    float m = 0, v = 0;
    for (int i = 0; i < D; i++) m += x[i];
    m /= D;
    for (int i = 0; i < D; i++) { float d = x[i] - m; v += d * d; }
    v = 1.0f / sqrtf(v / D + 1e-5f);
    for (int i = 0; i < D; i++) o[i] = (x[i] - m) * v * w[i] + b[i];
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

// ---- addresser state ----
struct Addr {
    int k, Dd, Mx;
    vector<float> coef, b;
    vector<int> lens, rows;   // rows[k*Mx]
};
vector<Addr> AD;

// row-major (per-row contiguous) copies of the down weights, built at load:
// Wd is stored [D][FFN], so the k-row gather strides 8 KB per element and
// misses cache on every read. WdRM[r*D+i] == Wd[i*FFN+r] makes each selected
// row one contiguous 2 KB scan the compiler can vectorize.
static vector<float*> WdRM;

static void load_addr(const char* fn) {
    ifstream f(fn, ios::binary);
    uint32_t nL; f.read((char*)&nL, 4);
    for (uint32_t l = 0; l < nL; l++) {
        Addr a; uint32_t k, dd, mx;
        f.read((char*)&k, 4); f.read((char*)&dd, 4); f.read((char*)&mx, 4);
        a.k = k; a.Dd = dd; a.Mx = mx;
        a.coef.resize((size_t)k * dd); a.b.resize(k);
        a.lens.resize(k); a.rows.resize((size_t)k * mx);
        f.read((char*)a.coef.data(), (size_t)k * dd * 4);
        f.read((char*)a.b.data(), k * 4);
        f.read((char*)a.lens.data(), k * 4);
        f.read((char*)a.rows.data(), (size_t)k * mx * 4);
        AD.push_back(move(a));
    }
}

// addressed FFN (exact mirror of the PyTorch port):
//   h = ln2(x); p=sigmoid(h@coef^T+b); walk clusters desc until C rows;
//   g/u = h@Wg[rows]^T / h@Wu[rows]^T (no bias); a=silu(g)*u;
//   top-KTOP by |a|; y = sum a_sel * Wd[:, sel] (no bias).
static void ffn_addr_tok(const int l, const float* x, float* y) {
    static float h[D], hs[FFN];
    ln(x, get("blocks." + to_string(l) + ".ln2.weight"), get("blocks." + to_string(l) + ".ln2.bias"), h);
    const Addr& a = AD[l];
    const float* Wg = get("blocks." + to_string(l) + ".gate.weight");
    const float* Wu = get("blocks." + to_string(l) + ".up.weight");
    const float* Wd = get("blocks." + to_string(l) + ".down.weight");
    // cluster probs
    vector<pair<float,int>> pr(a.k);
    for (int c = 0; c < a.k; c++) {
        const float* w = &a.coef[(size_t)c * D];
        float z = a.b[c];
        for (int i = 0; i < D; i++) z += h[i] * w[i];
        pr[c] = { 1.0f / (1.0f + expf(-z)), c };
    }
    sort(pr.begin(), pr.end(), [](auto& p, auto& q){ return p.first > q.first; });
    // candidate rows, then gate/up, then topk
    const int Cmax = min(CBUDGET, 2048);
    int cc[2048]; bool ok[2048]; float sc[2048];
    int C = 0;
    for (int i = 0; i < a.k && C < Cmax; i++) {
        int cid = pr[i].second, ln_ = a.lens[cid];
        for (int j = 0; j < ln_ && C < Cmax; j++)
            cc[C] = a.rows[(size_t)cid * a.Mx + j], ok[C] = true, C++;
    }
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < C; t++) {
        int r = cc[t];
        float g = 0, u = 0;
        const float* wr = Wg + (size_t)r * D; const float* wu = Wu + (size_t)r * D;
        for (int i = 0; i < D; i++) { g += h[i] * wr[i]; u += h[i] * wu[i]; }
        float av = silu(g) * u;
        hs[t] = av; sc[t] = fabsf(av);
    }
    // top-KTOP indices of sc
    vector<pair<float,int>> top;
    top.reserve(C);
    for (int t = 0; t < C; t++) top.push_back({ sc[t], t });
    if ((int)top.size() > KTOP)
        partial_sort(top.begin(), top.begin() + KTOP, top.end(),
                     [](auto& p, auto& q){ return p.first > q.first; });
    for (int i = 0; i < D; i++) y[i] = 0;
    int nt = min(KTOP, (int)top.size());
    for (int s = 0; s < nt; s++) {
        int r = cc[top[s].second];
        float av = hs[top[s].second];
        const float* wr = WdRM[l] + (size_t)r * D;
        for (int i = 0; i < D; i++) y[i] += av * wr[i];
    }
}

static void ffn_dense_tok(const int l, const float* x, float* y) {
    static float h[D], g[FFN], u[FFN];
    ln(x, get("blocks." + to_string(l) + ".ln2.weight"), get("blocks." + to_string(l) + ".ln2.bias"), h);
    const float* Wg = get("blocks." + to_string(l) + ".gate.weight");
    const float* bg = get("blocks." + to_string(l) + ".gate.bias");
    const float* Wu = get("blocks." + to_string(l) + ".up.weight");
    const float* bu = get("blocks." + to_string(l) + ".up.bias");
    const float* Wd = get("blocks." + to_string(l) + ".down.weight");
    const float* bd = get("blocks." + to_string(l) + ".down.bias");
    #pragma omp parallel for
    for (int r = 0; r < FFN; r++) {
        const float* wr = Wg + (size_t)r * D; const float* wu = Wu + (size_t)r * D;
        float gi = bg[r], ui = bu[r];
        for (int i = 0; i < D; i++) { gi += h[i] * wr[i]; ui += h[i] * wu[i]; }
        g[r] = silu(gi) * ui;
    }
    for (int i = 0; i < D; i++) y[i] = bd[i];
    for (int r = 0; r < FFN; r++) {
        float a = g[r];
        if (a == 0) continue;
        const float* wr = WdRM[l] + (size_t)r * D;
        for (int i = 0; i < D; i++) y[i] += a * wr[i];
    }
}

static void ffn_oracle_tok(const int l, const float* x, float* y) {
    static float h[D], g[FFN], u[FFN];
    ln(x, get("blocks." + to_string(l) + ".ln2.weight"), get("blocks." + to_string(l) + ".ln2.bias"), h);
    const float* Wg = get("blocks." + to_string(l) + ".gate.weight");
    const float* bg = get("blocks." + to_string(l) + ".gate.bias");
    const float* Wu = get("blocks." + to_string(l) + ".up.weight");
    const float* bu = get("blocks." + to_string(l) + ".up.bias");
    const float* Wd = get("blocks." + to_string(l) + ".down.weight");
    const float* bd = get("blocks." + to_string(l) + ".down.bias");
    #pragma omp parallel for
    for (int r = 0; r < FFN; r++) {
        const float* wr = Wg + (size_t)r * D; const float* wu = Wu + (size_t)r * D;
        float gi = bg[r], ui = bu[r];
        for (int i = 0; i < D; i++) { gi += h[i] * wr[i]; ui += h[i] * wu[i]; }
        g[r] = fabsf(silu(gi) * ui);
    }
    vector<pair<float,int>> top(FFN);
    for (int r = 0; r < FFN; r++) top[r] = { g[r], r };
    partial_sort(top.begin(), top.begin() + KTOP, top.end(),
                 [](auto& p, auto& q){ return p.first > q.first; });
    for (int i = 0; i < D; i++) y[i] = bd[i];
    for (int s = 0; s < KTOP; s++) {
        int r = top[s].second;
        const float* wr = Wg + (size_t)r * D; const float* wu = Wu + (size_t)r * D;
        float gi = bg[r], ui = bu[r];
        for (int i = 0; i < D; i++) { gi += h[i] * wr[i]; ui += h[i] * wu[i]; }
        float av = silu(gi) * ui;
        const float* wd = WdRM[l] + (size_t)r * D;
        for (int i = 0; i < D; i++) y[i] += av * wd[i];
    }
}

enum Mode { M_DENSE, M_ADDR, M_ORACLE };
Mode MODE = M_DENSE;

static void block_fwd(const int l, float* x, float* kvk, float* kvv, int pos) {
    static float hh[D], q[NH * HD], k[NH * HD], v[NH * HD], a[D], ff[D];
    ln(x, get("blocks." + to_string(l) + ".ln1.weight"), get("blocks." + to_string(l) + ".ln1.bias"), hh);
    const float* Wq = get("blocks." + to_string(l) + ".qkv.weight");
    const float* bq = get("blocks." + to_string(l) + ".qkv.bias");
    double _t0 = tsec();
    #pragma omp parallel for
    for (int i = 0; i < 3 * D; i++) {
        const float* w = Wq + (size_t)i * D;
        float z = bq[i];
        for (int j = 0; j < D; j++) z += hh[j] * w[j];
        if (i < D) q[i] = z;
        else if (i < 2 * D) k[i - D] = z;
        else v[i - 2 * D] = z;
    }
    T_QKV += tsec() - _t0;
    _t0 = tsec();
    memcpy(kvk + (size_t)pos * D, k, D * sizeof(float));
    memcpy(kvv + (size_t)pos * D, v, D * sizeof(float));
    const float* Wp = get("blocks." + to_string(l) + ".proj.weight");
    const float* bp = get("blocks." + to_string(l) + ".proj.bias");
    #pragma omp parallel for
    for (int h = 0; h < NH; h++) {
        float sc[SEQ], mx, sm;
        mx = -1e30f;
        for (int s = 0; s <= pos; s++) {
            float z = 0;
            for (int d = 0; d < HD; d++)
                z += q[h * HD + d] * kvk[(size_t)s * D + h * HD + d];
            z *= 1.0f / sqrtf((float)HD);
            sc[s] = z; if (z > mx) mx = z;
        }
        sm = 0;
        for (int s = 0; s <= pos; s++) { sc[s] = expf(sc[s] - mx); sm += sc[s]; }
        for (int d = 0; d < HD; d++) {
            float acc = 0;
            for (int s = 0; s <= pos; s++)
                acc += (sc[s] / sm) * kvv[(size_t)s * D + h * HD + d];
            a[h * HD + d] = acc;
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < D; i++) {
        const float* w = Wp + (size_t)i * D;
        float z = bp[i];
        for (int j = 0; j < D; j++) z += a[j] * w[j];
        x[i] += z;
    }
    T_ATTN += tsec() - _t0;
    double _tf = tsec();
    if (MODE == M_DENSE) ffn_dense_tok(l, x, ff);
    else if (MODE == M_ADDR) ffn_addr_tok(l, x, ff);
    else ffn_oracle_tok(l, x, ff);
    T_FFN += tsec() - _tf;
    for (int i = 0; i < D; i++) x[i] += ff[i];
}

static float logit(const float* x, int t) {
    const float* E = get("emb.weight");
    const float* w = E + (size_t)t * D;
    float z = 0;
    for (int i = 0; i < D; i++) z += x[i] * w[i];
    return z;
}

int main(int argc, char** argv) {
    const char* hfn = nullptr, *afn = nullptr, *pfn = nullptr, *vfn = nullptr;
    const char* cmd = nullptr; int N = 128;
    for (int i = 1; i < argc; i++) {
        string s = argv[i];
        if (s == "--mode") { string m = argv[++i];
            MODE = m == "addr" ? M_ADDR : m == "oracle" ? M_ORACLE : M_DENSE; }
        else if (s == "--hyper") hfn = argv[++i];
        else if (s == "--addr") afn = argv[++i];
        else if (s == "--prompt") pfn = argv[++i];
        else if (s == "--val") vfn = argv[++i];
        else if (s == "bpt") cmd = "bpt";
        else if (s == "gen") cmd = "gen";
        else if (s == "dbg") cmd = "dbg";
        else if (s == "-C") CBUDGET = atoi(argv[++i]);
        else if (s == "-n") N = atoi(argv[++i]);
    }
    if (!hfn) { fprintf(stderr, "usage: p3a_engine --hyper m.hyper [--addr a.bin] [--mode dense|addr|oracle] bpt --val ids.bin | gen --prompt p.txt -n N\n"); return 1; }

    ifstream f(hfn, ios::binary);
    char magic[4]; f.read(magic, 4);
    if (strncmp(magic, "HYP2", 4)) { fprintf(stderr, "bad magic\n"); return 1; }
    uint32_t vocab, nt; f.read((char*)&vocab, 4); f.read((char*)&nt, 4);
    V = vocab;
    vector<T> raw(nt);
    for (uint32_t i = 0; i < nt; i++) {
        T& t = raw[i];
        uint32_t nl; f.read((char*)&nl, 4);
        string name(nl, 0); f.read(&name[0], nl);
        uint32_t type, nd; f.read((char*)&type, 4); f.read((char*)&nd, 4);
        uint32_t el = 1;
        for (uint32_t d = 0; d < nd; d++) { uint32_t sd; f.read((char*)&sd, 4); t.shape.push_back(sd); el *= sd; }
        t.data.resize(el); f.read((char*)t.data.data(), el * 4);
        M[name] = &t;
    }
    fprintf(stderr, "[p3a] loaded %u tensors, V=%d, mode=%s\n", nt, V,
            MODE == M_DENSE ? "dense" : MODE == M_ADDR ? "addr" : "oracle");
    if (afn) load_addr(afn);

    // build the row-major down-weight copies (one-time, ~33 MB)
    for (int l = 0; l < L; l++) {
        const float* Wd = get("blocks." + to_string(l) + ".down.weight");
        float* rm = (float*)malloc(sizeof(float) * (size_t)D * FFN);
        for (int r = 0; r < FFN; r++)
            for (int i = 0; i < D; i++) rm[(size_t)r * D + i] = Wd[(size_t)i * FFN + r];
        WdRM.push_back(rm);
    }

    if (cmd && string(cmd) == "dbg" && pfn) {
        vector<int> toks;
        { ifstream pf(pfn); string s; getline(pf, s);
          size_t p = 0;
          while (p < s.size()) {
              size_t q = s.find(',', p);
              if (q == string::npos) q = s.size();
              toks.push_back(atoi(s.substr(p, q - p).c_str()));
              p = q + 1;
          } }
        int tok = toks[0];
        vector<float> kvk((size_t)L * SEQ * D), kvv((size_t)L * SEQ * D);
        static float x[D], xo[D];
        const float* E = get("emb.weight");
        const float* P = get("pos.weight");
        for (int i = 0; i < D; i++) x[i] = E[(size_t)tok * D + i] + P[i];
        printf("x0:"); for (int i = 0; i < 4; i++) printf(" %.6f", x[i]); printf("\n");
        for (int l = 0; l < L; l++) {
            block_fwd(l, x, &kvk[(size_t)l * SEQ * D], &kvv[(size_t)l * SEQ * D], 0);
            printf("x%d:", l + 1); for (int i = 0; i < 4; i++) printf(" %.6f", x[i]); printf("\n");
        }
        printf("xL:"); for (int i = 0; i < 4; i++) printf(" %.6f", x[i]); printf("\n");
        ln(x, get("ln.weight"), get("ln.bias"), xo);
        printf("ln:"); for (int i = 0; i < 4; i++) printf(" %.6f", xo[i]); printf("\n");
        for (int t : {1854, 1013, 5734, 209}) printf("lg[%d]=%.6f\n", t, logit(xo, t));
        return 0;
    }
    if (cmd && string(cmd) == "bpt" && vfn) {
        ifstream vf(vfn, ios::binary);
        int32_t n; vf.read((char*)&n, 4);
        vector<int> val(n);
        vf.read((char*)val.data(), (size_t)n * 4);
        fprintf(stderr, "[p3a] val tokens: %d\n", n);
        const int NB = 16;
        vector<float> kvk((size_t)L * SEQ * D), kvv((size_t)L * SEQ * D);
        static float x[D], xo[D];
        double tot = 0; int cnt = 0;
        auto t0 = Clock::now();
        for (int w = 0; w < NB; w++) {
            memset(kvk.data(), 0, kvk.size() * 4);
            memset(kvv.data(), 0, kvv.size() * 4);
            for (int pos = 0; pos < SEQ; pos++) {
                int tok = val[w * SEQ + pos];
                const float* E = get("emb.weight");
                const float* P = get("pos.weight");
                for (int i = 0; i < D; i++) x[i] = E[(size_t)tok * D + i] + P[(size_t)pos * D + i];
                for (int l = 0; l < L; l++)
                    block_fwd(l, x, &kvk[(size_t)l * SEQ * D], &kvv[(size_t)l * SEQ * D], pos);
                ln(x, get("ln.weight"), get("ln.bias"), xo);
                if (pos < SEQ - 1) {
                    int tgt = val[w * SEQ + pos + 1];
                    static vector<float> lg;
                    lg.resize(V);
                    const float* E2 = get("emb.weight");
                    #pragma omp parallel for
                    for (int t = 0; t < V; t++) {
                        const float* w2 = E2 + (size_t)t * D;
                        float z = 0;
                        for (int i = 0; i < D; i++) z += xo[i] * w2[i];
                        lg[t] = z;
                    }
                    float ls = *max_element(lg.begin(), lg.end());
                    double se = 0;
                    for (int t = 0; t < V; t++) se += expf(lg[t] - ls);
                    float lgt = lg[tgt] - ls - logf((float)se);
                    tot += -lgt / log(2.0); cnt++;
                }
            }
        }
        double dt = chrono::duration<double>(Clock::now() - t0).count();
        printf("bpt=%.4f  (%d targets, %.1fs)\n", tot / cnt, cnt, dt);
        return 0;
    }

    if (cmd && string(cmd) == "gen" && pfn) {
        vector<int> toks;
        { ifstream pf(pfn); string s; getline(pf, s);
          size_t p = 0;
          while (p < s.size()) {
              size_t q = s.find(',', p);
              if (q == string::npos) q = s.size();
              toks.push_back(atoi(s.substr(p, q - p).c_str()));
              p = q + 1;
          } }
        vector<float> kvk((size_t)L * SEQ * D), kvv((size_t)L * SEQ * D);
        static float x[D], xo[D];
        const float* E = get("emb.weight");
        const float* P = get("pos.weight");
        int PLEN = (int)toks.size();
        auto t0 = Clock::now();
        for (int pos = 0; pos < PLEN + N; pos++) {
            int tok = pos < PLEN ? toks[pos] : toks.back();
            if (pos == 0 || pos >= PLEN) {
                for (int i = 0; i < D; i++) x[i] = E[(size_t)tok * D + i] + P[(size_t)pos * D + i];
                for (int l = 0; l < L; l++)
                    block_fwd(l, x, &kvk[(size_t)l * SEQ * D], &kvv[(size_t)l * SEQ * D], pos);
            } else {
                for (int i = 0; i < D; i++) x[i] = E[(size_t)tok * D + i] + P[(size_t)pos * D + i];
                for (int l = 0; l < L; l++)
                    block_fwd(l, x, &kvk[(size_t)l * SEQ * D], &kvv[(size_t)l * SEQ * D], pos);
            }
            ln(x, get("ln.weight"), get("ln.bias"), xo);
            if (pos >= PLEN - 1) {
                float best = -1e30f; int bi = 0;
                double _tl = tsec();
                { static vector<float> _lg; _lg.resize(V);
                  #pragma omp parallel for
                  for (int t = 0; t < V; t++) _lg[t] = logit(xo, t);
                  for (int t = 0; t < V; t++) if (_lg[t] > best) { best = _lg[t]; bi = t; } }
                T_LOGIT += tsec() - _tl;
                printf("%d\n", bi);
                toks.push_back(bi);
            }
        }
        double dt = chrono::duration<double>(Clock::now() - t0).count();
        fprintf(stderr, "[p3a] gen %d tokens in %.3fs -> %.1f tok/s (B=1)\n", N, dt, N / dt);
        {
            double tot = dt;
            fprintf(stderr, "[prof] ms/tok over %d steps: qkv=%.2f attn+proj=%.2f ffn=%.2f logits=%.2f other=%.2f | per-layer ms: qkv=%.2f attn+proj=%.2f ffn=%.2f\n",
                    PLEN + N, 1e3*T_QKV/(PLEN+N), 1e3*T_ATTN/(PLEN+N),
                    1e3*T_FFN/(PLEN+N), 1e3*T_LOGIT/(PLEN+N),
                    1e3*(tot - T_QKV - T_ATTN - T_PROJ - T_FFN - T_LOGIT)/(PLEN+N),
                    1e3*T_QKV/(PLEN+N)/L, 1e3*T_ATTN/(PLEN+N)/L, 1e3*T_FFN/(PLEN+N)/L);
        }
        return 0;
    }
    fprintf(stderr, "nothing to do\n");
    return 1;
}


