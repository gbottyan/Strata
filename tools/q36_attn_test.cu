// Standalone check of k_attn_part's int8 K path against a CPU reference.
// The kernel body is copied verbatim from src/qwen35/q36_engine.cu, so what is tested is what the engine runs.
// The kernel body is regenerated from the engine, so keep it in sync after editing k_attn_part:
//   sed -n '360,446p' src/qwen35/q36_engine.cu   (the kernel, from its template line to its closing brace)
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
    float2* const* ksc = nullptr;
    float2* const* vsc = nullptr;
};

// ---- the kernel, verbatim from the engine ----
template <bool Q8>
__global__ void k_attn_part(const float* q, const float* kc, const float* vc, int8_t* kc8, int8_t* vc8,
                            float2* ksc, float2* vsc, const int32_t* pos0, int t0, int H,
                            int HKV, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl, RowSel rs) {
    constexpr int HD = 256;
    __shared__ float qs[HD], sc[kChunkCells], sh[kThreads];
    // The head is the grid's fast axis (the launch site pairs it with blockIdx.x).  This was tried as a fix for
    // decode-at-depth: the H/HKV = 8 query heads of one KV head read the same K and V, blocks launch x-first, and the
    // 2 MB of L2 on a Pascal card is far too small to bridge 375 blocks, so the layout looked like 8x the traffic.
    // It measured neutral (attn 3851 ms vs 3813 ms over 25 steps at 96k), so it is not the cost.  Kept because the
    // coalesced K read below wants blocks of one head adjacent, and it costs nothing.
    const int row = blockIdx.x, c = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
        if (Q8) {
            kc8 = rs.kc8[rs.slot[t]];
            vc8 = rs.vc8[rs.slot[t]];
            ksc = rs.ksc[rs.slot[t]];
            vsc = rs.vsc[rs.slot[t]];
        }
    }
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int j = threadIdx.x;
    qs[j] = q[((size_t) t * H + h) * HD + j];
    __syncthreads();
    // sum of the query's dims: the affine dequantisation pulls out of the K dot product,
    //   sum_i q[i] * (code[i] * s + m)  ==  s * sum_i q[i] * code[i]  +  m * sum_i q[i]
    const float sumq = Q8 ? block_sum(qs[j], sh) : 0.f;
    // K's dot products, one warp per cell.  The one-thread-per-cell layout that the softmax and the V sum below use
    // walks a cell's 256 dims in that thread, so a warp's 32 lanes read 32 rows 256 bytes apart: 32 sectors fetched for
    // 32 bytes used, 32x the traffic.  Measured at 96k context: 154 ms of a 189 ms decode step.  A warp taking one
    // cell reads its 256 contiguous bytes (8 dims per lane) in 8 full sectors; the block's 8 warps take 8 cells a pass.
    const int warp = j >> 5, lane = j & 31;
    for (int it = 0; it < kChunkCells / 8; ++it) {
        const int cell = it * 8 + warp, pc = p0 + cell;
        float a = 0.f;
        if (pc < npos) {
            if (Q8) {
                const int8_t* kp = kc8 + ((size_t) kvh * ctx + pc) * HD + lane * 8;
#pragma unroll
                for (int i = 0; i < 8; ++i) a += qs[lane * 8 + i] * (float) kp[i];
                a *= ksc[(size_t) kvh * ctx + pc].x;   // the cell's scale is uniform, so it factors out of the warp
            } else {
                const float* kp = kc + ((size_t) kvh * ctx + pc) * HD + lane * 8;
#pragma unroll
                for (int i = 0; i < 8; ++i) a += qs[lane * 8 + i] * kp[i];
            }
        }
        const float dot = warp_sum(a);   // every lane calls it: a shuffle needs the whole warp
        // The affine dequant's offset is a constant of the cell, so it goes in once, after the reduction.  Adding it to
        // each lane's partial (as the one-thread-per-cell layout did) counts it 32 times: a per-cell error of
        // 31 * sumq * scale.y * scale, up to 0.025 on a score, which exp() turns into 2.6% on the attention output -
        // enough to make generation collapse into repetition.  tools/attn_test.cu checks this kernel against a CPU
        // reference; that error is what it caught, and the same layout now passes at 3.8e-07 relative.
        if (lane == 0) sc[cell] = Q8 ? (dot + sumq * ksc[(size_t) kvh * ctx + pc].y) * scale : dot * scale;
    }
    __syncthreads();
    const int p = p0 + j;
    const float s = p < npos ? sc[j] : -INFINITY;
    const float m = block_max(s, sh);
    const float e = p < npos ? expf(s - m) : 0.f;
    sc[j] = e;
    const float l = block_sum(e, sh);
    const int n = min(kChunkCells, npos - p0);
    float acc = 0.f;
    if (Q8) {
        const int8_t* vp = vc8 + ((size_t) kvh * ctx + p0) * HD + j;
        const float2* vsr = vsc + (size_t) kvh * ctx + p0;
        for (int k = 0; k < n; ++k) {
            const float2 p2 = vsr[k];
            acc += sc[k] * ((float) vp[(size_t) k * HD] * p2.x + p2.y);
        }
    } else {
        const float* vp = vc + ((size_t) kvh * ctx + p0) * HD + j;
        for (int k = 0; k < n; ++k) acc += sc[k] * vp[(size_t) k * HD];
    }
    const size_t o = (size_t) row * n_chunk + c;
    pacc[o * HD + j] = acc;
    if (j == 0) {
        pm[o] = m;
        pl[o] = l;
    }
}
// ---- the test ----
static unsigned long long lcg = 88172645463325252ULL;
static double rnd() { lcg ^= lcg << 13; lcg ^= lcg >> 7; lcg ^= lcg << 17; return (double) (lcg >> 11) / 9007199254740992.0; }

int main() {
    const int HD = 256, H = 16, HKV = 2, ctx = 1024, n_chunk = ctx / kChunkCells;
    const int pos0 = 899;                       // row 0 sees cells [0, 900)
    const int npos = pos0 + 1;
    const float scale = 1.0f / sqrtf((float) HD);

    std::vector<float> q((size_t) H * HD);
    std::vector<int8_t> kc8((size_t) HKV * ctx * HD), vc8((size_t) HKV * ctx * HD);
    std::vector<float2> ksc((size_t) HKV * ctx), vsc((size_t) HKV * ctx);
    for (auto& f : q) f = (float) (rnd() * 2 - 1);
    for (auto& b : kc8) b = (int8_t) (int) (rnd() * 255 - 127);
    for (auto& b : vc8) b = (int8_t) (int) (rnd() * 255 - 127);
    for (auto& p : ksc) { p.x = (float) (0.001 + rnd() * 0.009); p.y = (float) (rnd() * 0.02 - 0.01); }
    for (auto& p : vsc) { p.x = (float) (0.001 + rnd() * 0.009); p.y = (float) (rnd() * 0.02 - 0.01); }

    std::vector<float> hpacc((size_t) n_chunk * HD, 0.f), hpm(n_chunk, 0.f), hpl(n_chunk, 0.f);
    std::vector<float> ref_acc((size_t) n_chunk * HD, 0.f), ref_m(n_chunk, 0.f), ref_l(n_chunk, 0.f);

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
    cudaMalloc(&d_pm, n_chunk * 4);
    cudaMalloc(&d_pl, n_chunk * 4);
    cudaMemset(d_pacc, 0, hpacc.size() * 4);
    cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kc8, kc8.data(), kc8.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vc8, vc8.data(), vc8.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ksc, ksc.data(), ksc.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vsc, vsc.data(), vsc.size() * 8, cudaMemcpyHostToDevice);
    cudaMalloc(&d_pos0, 4);
    cudaMemcpy(d_pos0, &pos0, 4, cudaMemcpyHostToDevice);

    k_attn_part<true><<<dim3(1, n_chunk), kThreads>>>(d_q, nullptr, nullptr, d_kc8, d_vc8, d_ksc, d_vsc, d_pos0, 0,
                                                      H, HKV, ctx, scale, n_chunk, d_pacc, d_pm, d_pl, RowSel{});
    std::printf("launch error: %s\n", cudaGetErrorString(cudaGetLastError()));
    std::printf("sync: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    int32_t chk = -1;
    cudaMemcpy(&chk, d_pos0, 4, cudaMemcpyDeviceToHost);
    std::printf("pos0 on device = %d\n", chk);
    cudaMemcpy(hpacc.data(), d_pacc, hpacc.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hpm.data(), d_pm, n_chunk * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hpl.data(), d_pl, n_chunk * 4, cudaMemcpyDeviceToHost);

    // the reference: the same arithmetic, in the kernels order
    float sumq = 0.f;
    for (int i = 0; i < HD; ++i) sumq += q[i];
    for (int c = 0; c < n_chunk; ++c) {
        const int p0 = c * kChunkCells;
        if (p0 >= npos) continue;
        float sc[kChunkCells], m = -1e30f;
        for (int cell = 0; cell < kChunkCells; ++cell) {
            const int p = p0 + cell;
            if (p >= npos) { sc[cell] = -1e30f; continue; }
            float a = 0.f;
            for (int i = 0; i < HD; ++i) a += q[i] * (float) kc8[((size_t) 0 * ctx + p) * HD + i];
            const float2 ks = ksc[(size_t) 0 * ctx + p];
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
                const float2 vs = vsc[(size_t) 0 * ctx + p0 + k];
                acc += sc[k] * ((float) vc8[((size_t) 0 * ctx + p0 + k) * HD + d] * vs.x + vs.y);
            }
            ref_acc[(size_t) c * HD + d] = acc;
        }
        ref_m[c] = m;
        ref_l[c] = l;
    }

    double em = 0, ea = 0, el = 0, am = 0, aa = 0, al = 0;
    for (int c = 0; c < n_chunk; ++c) {
        em = std::max(em, (double) std::fabs(hpm[c] - ref_m[c]));
        am = std::max(am, (double) std::fabs(ref_m[c]));
        el = std::max(el, (double) std::fabs(hpl[c] - ref_l[c]));
        al = std::max(al, (double) std::fabs(ref_l[c]));
        for (int d = 0; d < HD; ++d) {
            ea = std::max(ea, (double) std::fabs(hpacc[(size_t) c * HD + d] - ref_acc[(size_t) c * HD + d]));
            aa = std::max(aa, (double) std::fabs(ref_acc[(size_t) c * HD + d]));
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
