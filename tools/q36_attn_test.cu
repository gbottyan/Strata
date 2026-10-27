// Standalone check of k_attn_part's int8 K path against a CPU reference.
// The kernel body is copied verbatim from src/qwen35/q36_engine.cu, so what is tested is what the engine runs.
// The kernel body is regenerated from the engine, so keep it in sync after editing k_attn_part:
//   sed -n '375,505p' src/qwen35/q36_engine.cu   (the kernel, from its template line to its closing brace)
// Build and run on the box (it needs the GPU to itself: a second CUDA context will not start next to a running
// engine):  nvcc -std=c++20 -arch=sm_61 -O2 -o q36_attn_test q36_attn_test.cu && ./q36_attn_test
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

constexpr int kThreads = 256;
constexpr int kChunkCells = 256;

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
struct RowSel {
    const int32_t* cell = nullptr;
    const int32_t* slot = nullptr;
    const int32_t* delta = nullptr;
    float* const* kc = nullptr;
    float* const* vc = nullptr;
    int8_t* const* kc8 = nullptr;
    int8_t* const* vc8 = nullptr;
    int8_t* const* kc8h = nullptr;
    int8_t* const* vc8h = nullptr;
    float2* const* ksc = nullptr;
    float2* const* vsc = nullptr;
};

// ---- the kernel, verbatim from the engine ----
template <bool Q8, int GQ>
__global__ void k_attn_part(const float* q, const float* kc, const float* vc, int8_t* kc8, int8_t* vc8,
                            int8_t* kc8h, int8_t* vc8h,
                            float2* ksc, float2* vsc, const int32_t* pos0, int t0, int H,
                            int HKV, int ctx, int nres, float scale, int n_chunk, float* pacc, float* pm, float* pl,
                            RowSel rs) {
    constexpr int HD = 256;
    __shared__ float qs[GQ][HD], sc[GQ][kChunkCells], sh[kThreads], sumq[GQ];
    const int row_g = blockIdx.x, c = blockIdx.y;
    const int G = H / HKV, GPG = G / GQ;   // query heads per KV head, groups of them per KV head
    const int t = t0 + row_g / (H / GQ), hg = row_g % (H / GQ);
    const int kvh = hg / GPG, g0 = (hg % GPG) * GQ;
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
        if (Q8) {
            kc8 = rs.kc8[rs.slot[t]];
            vc8 = rs.vc8[rs.slot[t]];
            ksc = rs.ksc[rs.slot[t]];
            vsc = rs.vsc[rs.slot[t]];
            if (rs.kc8h) { kc8h = rs.kc8h[rs.slot[t]]; vc8h = rs.vc8h[rs.slot[t]]; }
        }
    }
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    // --kv-resident: nres is a multiple of kChunkCells, so this chunk lies wholly on one side of the cut and one base
    // and one stride serve it.  The scales stay in one table at stride ctx, in VRAM, read in the same loops.
    const bool res = p0 < nres;
    const int stride = res ? nres : ctx - nres, off = res ? 0 : nres;
    const int j = threadIdx.x;
    for (int g = 0; g < GQ; ++g) qs[g][j] = q[((size_t) t * H + kvh * G + g0 + g) * HD + j];
    __syncthreads();
    // sum of the query's dims: the affine dequantisation pulls out of the K dot product,
    //   sum_i q[i] * (code[i] * s + m)  ==  s * sum_i q[i] * code[i]  +  m * sum_i q[i]
    if (Q8) {
        for (int g = 0; g < GQ; ++g) sumq[g] = block_sum(qs[g][j], sh);
        __syncthreads();
    }
    // K's dot products, one warp per cell.  The one-thread-per-cell layout that the softmax and the V sum below use
    // walks a cell's 256 dims in that thread, so a warp's 32 lanes read 32 rows 256 bytes apart: 32 sectors fetched for
    // 32 bytes used, 32x the traffic.  A warp taking one cell reads its 256 contiguous bytes (8 dims per lane) in 8
    // full sectors; the block's 8 warps take 8 cells a pass, and each cell's row serves all GQ heads of the group.
    const int warp = j >> 5, lane = j & 31;
    for (int it = 0; it < kChunkCells / 8; ++it) {
        const int cell = it * 8 + warp, pc = p0 + cell;
        if (pc >= npos) break;   // warp-uniform: cell is a function of the warp, so the whole warp leaves together
        if (Q8) {
            const int8_t* kp = (res ? kc8 : kc8h) + ((size_t) kvh * stride + (pc - off)) * HD + lane * 8;
            const float2 k2 = ksc[(size_t) kvh * ctx + pc];
            float a[GQ];
#pragma unroll
            for (int g = 0; g < GQ; ++g) {
                a[g] = 0.f;
#pragma unroll
                for (int i = 0; i < 8; ++i) a[g] += qs[g][lane * 8 + i] * (float) kp[i];
                a[g] *= k2.x;   // the cell's scale is uniform, so it factors out of the warp
            }
#pragma unroll
            for (int g = 0; g < GQ; ++g) a[g] = warp_sum(a[g]);   // every lane calls it: a shuffle needs the whole warp
            if (lane == 0) {
#pragma unroll
                for (int g = 0; g < GQ; ++g) {
                    // The affine dequant's offset is a constant of the cell, so it goes in once, after the reduction.
                    // Adding it to each lane's partial (as the one-thread-per-cell layout did) counts it 32 times: a
                    // per-cell error of 31 * sumq * scale.y * scale, up to 0.025 on a score, which exp() turns into
                    // 2.6% on the attention output - enough to make generation collapse into repetition.
                    sc[g][cell] = (a[g] + sumq[g] * k2.y) * scale;
                }
            }
        } else {
            const float* kp = kc + ((size_t) kvh * ctx + pc) * HD + lane * 8;
            float a[GQ];
#pragma unroll
            for (int g = 0; g < GQ; ++g) {
                a[g] = 0.f;
#pragma unroll
                for (int i = 0; i < 8; ++i) a[g] += qs[g][lane * 8 + i] * kp[i];
                a[g] = warp_sum(a[g]);
            }
            if (lane == 0) {
#pragma unroll
                for (int g = 0; g < GQ; ++g) sc[g][cell] = a[g] * scale;
            }
        }
    }
    __syncthreads();
    // the causal softmax, per head over the chunk's cells
    const int p = p0 + j;
    float m[GQ], l[GQ];
    for (int g = 0; g < GQ; ++g) {
        const float s = p < npos ? sc[g][j] : -INFINITY;
        m[g] = block_max(s, sh);
        const float e = p < npos ? expf(s - m[g]) : 0.f;
        sc[g][j] = e;
        l[g] = block_sum(e, sh);
    }
    __syncthreads();
    // V's weighted sum: thread j owns dim j of the output and walks the chunk's cells; one dequantisation serves all
    // GQ heads, which is the other half of what sharing a block buys
    const int n = min(kChunkCells, npos - p0);
    float acc[GQ];
#pragma unroll
    for (int g = 0; g < GQ; ++g) acc[g] = 0.f;
    if (Q8) {
        const int8_t* vp = (res ? vc8 : vc8h) + ((size_t) kvh * stride + (p0 - off)) * HD + j;
        const float2* vsr = vsc + (size_t) kvh * ctx + p0;
        for (int k = 0; k < n; ++k) {
            const float2 p2 = vsr[k];
            const float v = (float) vp[(size_t) k * HD] * p2.x + p2.y;
#pragma unroll
            for (int g = 0; g < GQ; ++g) acc[g] += sc[g][k] * v;
        }
    } else {
        const float* vp = vc + ((size_t) kvh * ctx + p0) * HD + j;
        for (int k = 0; k < n; ++k) {
            const float v = vp[(size_t) k * HD];
#pragma unroll
            for (int g = 0; g < GQ; ++g) acc[g] += sc[g][k] * v;
        }
    }
#pragma unroll
    for (int g = 0; g < GQ; ++g) {
        const size_t o = ((size_t) t * H + kvh * G + g0 + g) * n_chunk + c;
        pacc[o * HD + j] = acc[g];
        if (j == 0) {
            pm[o] = m[g];
            pl[o] = l[g];
        }
    }
}
// ---- the test ----
static unsigned long long lcg = 88172645463325252ULL;
static double rnd() { lcg ^= lcg << 13; lcg ^= lcg >> 7; lcg ^= lcg << 17; return (double) (lcg >> 11) / 9007199254740992.0; }

int main() {
    constexpr int GQ = 2;
    const int HD = 256, H = 16, HKV = 2, ctx = 1024, n_chunk = ctx / kChunkCells;
    const int pos0 = 899;                       // every row sees cells [0, 900)
    const int npos = pos0 + 1;
    const float scale = 1.0f / sqrtf((float) HD);
    const int G = H / HKV;

    std::vector<float> q((size_t) H * HD);
    std::vector<int8_t> kc8((size_t) HKV * ctx * HD), vc8((size_t) HKV * ctx * HD);
    std::vector<float2> ksc((size_t) HKV * ctx), vsc((size_t) HKV * ctx);
    for (auto& f : q) f = (float) (rnd() * 2 - 1);
    for (auto& b : kc8) b = (int8_t) (int) (rnd() * 255 - 127);
    for (auto& b : vc8) b = (int8_t) (int) (rnd() * 255 - 127);
    for (auto& p : ksc) { p.x = (float) (0.001 + rnd() * 0.009); p.y = (float) (rnd() * 0.02 - 0.01); }
    for (auto& p : vsc) { p.x = (float) (0.001 + rnd() * 0.009); p.y = (float) (rnd() * 0.02 - 0.01); }

    const size_t nrow = (size_t) H * n_chunk;
    std::vector<float> hpacc(nrow * HD, 0.f), hpm(nrow, 0.f), hpl(nrow, 0.f);
    std::vector<float> ref_acc(nrow * HD, 0.f), ref_m(nrow, 0.f), ref_l(nrow, 0.f);

    float *d_q, *d_pacc, *d_pm, *d_pl;
    int8_t *d_kc8, *d_vc8;
    float2 *d_ksc, *d_vsc;
    int32_t* d_pos0 = nullptr;
    cudaMalloc(&d_q, q.size() * 4);
    cudaMalloc(&d_kc8, kc8.size());
    cudaMalloc(&d_vc8, vc8.size());
    cudaMalloc(&d_ksc, ksc.size() * 8);
    cudaMalloc(&d_vsc, vsc.size() * 8);
    cudaMalloc(&d_pacc, hpacc.size() * 4);
    cudaMalloc(&d_pm, nrow * 4);
    cudaMalloc(&d_pl, nrow * 4);
    cudaMemset(d_pacc, 0, hpacc.size() * 4);
    cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kc8, kc8.data(), kc8.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vc8, vc8.data(), vc8.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ksc, ksc.data(), ksc.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vsc, vsc.data(), vsc.size() * 8, cudaMemcpyHostToDevice);
    cudaMalloc(&d_pos0, 4);
    cudaMemcpy(d_pos0, &pos0, 4, cudaMemcpyHostToDevice);

    k_attn_part<true, GQ><<<dim3(H / GQ, n_chunk), kThreads>>>(d_q, nullptr, nullptr, d_kc8, d_vc8, nullptr, nullptr,
                                                               d_ksc, d_vsc, d_pos0, 0, H, HKV, ctx, ctx, scale,
                                                               n_chunk, d_pacc, d_pm, d_pl, RowSel{});
    std::printf("launch error: %s\n", cudaGetErrorString(cudaGetLastError()));
    std::printf("sync: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    cudaMemcpy(hpacc.data(), d_pacc, hpacc.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hpm.data(), d_pm, nrow * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hpl.data(), d_pl, nrow * 4, cudaMemcpyDeviceToHost);

    // the reference: the same arithmetic, in the kernels order, for every head
    for (int h = 0; h < H; ++h) {
        const int kvh = h / G;
        const float* qh = q.data() + (size_t) h * HD;
        float sumq = 0.f;
        for (int i = 0; i < HD; ++i) sumq += qh[i];
        for (int c = 0; c < n_chunk; ++c) {
            const int p0 = c * kChunkCells;
            if (p0 >= npos) continue;
            const size_t o = (size_t) h * n_chunk + c;
            float sc[kChunkCells], m = -1e30f;
            for (int cell = 0; cell < kChunkCells; ++cell) {
                const int p = p0 + cell;
                if (p >= npos) { sc[cell] = -1e30f; continue; }
                float a = 0.f;
                for (int i = 0; i < HD; ++i) a += qh[i] * (float) kc8[((size_t) kvh * ctx + p) * HD + i];
                const float2 ks = ksc[(size_t) kvh * ctx + p];
                sc[cell] = (a * ks.x + sumq * ks.y) * scale;
                if (sc[cell] > m) m = sc[cell];
            }
            float l = 0.f;
            for (int cell = 0; cell < kChunkCells; ++cell) {
                if (p0 + cell >= npos) { sc[cell] = 0.f; continue; }
                sc[cell] = expf(sc[cell] - m);
                l += sc[cell];
            }
            const int n = kChunkCells < npos - p0 ? kChunkCells : npos - p0;
            for (int d = 0; d < HD; ++d) {
                float acc = 0.f;
                for (int k = 0; k < n; ++k) {
                    const float2 vs = vsc[(size_t) kvh * ctx + p0 + k];
                    acc += sc[k] * ((float) vc8[((size_t) kvh * ctx + p0 + k) * HD + d] * vs.x + vs.y);
                }
                ref_acc[o * HD + d] = acc;
            }
            ref_m[o] = m;
            ref_l[o] = l;
        }
    }

    double em = 0, ea = 0, el = 0, am = 0, aa = 0, al = 0;
    for (size_t o = 0; o < nrow; ++o) {
        em = std::max(em, (double) std::fabs(hpm[o] - ref_m[o]));
        am = std::max(am, (double) std::fabs(ref_m[o]));
        el = std::max(el, (double) std::fabs(hpl[o] - ref_l[o]));
        al = std::max(al, (double) std::fabs(ref_l[o]));
        for (int d = 0; d < HD; ++d) {
            ea = std::max(ea, (double) std::fabs(hpacc[o * HD + d] - ref_acc[o * HD + d]));
            aa = std::max(aa, (double) std::fabs(ref_acc[o * HD + d]));
        }
    }
    std::printf("chunk max: kernel %.6f ref %.6f  (diff %.3e)\n", hpm[0], ref_m[0], em);
    std::printf("chunk sum: kernel %.6f ref %.6f  (diff %.3e)\n", hpl[0], ref_l[0], el);
    std::printf("acc: max |kernel - ref| = %.3e  (largest ref %.4f, relative %.3e)\n", ea, aa, aa > 0 ? ea / aa : 0.0);
    for (int c = 0; c < n_chunk; ++c)
        std::printf("  chunk %d: m %.5f/%.5f  l %.5f/%.5f  acc[0] %.6f/%.6f  acc[255] %.6f/%.6f\n", c, hpm[c],
                    ref_m[c], hpl[c], ref_l[c], hpacc[(size_t) c * HD], ref_acc[(size_t) c * HD],
                    hpacc[(size_t) c * HD + 255], ref_acc[(size_t) c * HD + 255]);
    const bool ok = ea < 1e-3 * (aa > 0 ? aa : 1) && el < 1e-3 * al && em < 1e-3 * (am > 0 ? am : 1);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
