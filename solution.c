
// src/q15_axpy_challenge.c
// Single-solution RVV challenge: Q15 y = a + alpha * b  (saturating to Q15)
//

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// -------------------- Scalar reference (no intrinsics) --------------------
static inline int16_t sat_q15_scalar(int32_t v)
{
    if (v > 32767)
        return 32767;
    if (v < -32768)
        return -32768;
    return (int16_t)v;
}

void q15_axpy_ref(const int16_t *a, const int16_t *b,
                  int16_t *y, int n, int16_t alpha)
{
    for (int i = 0; i < n; ++i)
    {
        int32_t acc = (int32_t)a[i] + (int32_t)alpha * (int32_t)b[i];
        y[i] = sat_q15_scalar(acc);
    }
}

// -------------------- RVV include per ratified v1.0 spec ------------------
#if __riscv_v_intrinsic >= 1000000
#include <riscv_vector.h> // v1.0 test macro & header inclusion
#endif

void q15_axpy_rvv(const int16_t *a, const int16_t *b,
                  int16_t *y, int n, int16_t alpha)
{
#if !defined(__riscv_vector)
    q15_axpy_ref(a, b, y, n, alpha);
#else
size_t vl;
    for (; n > 0; n -= vl) {
        // e16m4 allows the 32-bit intermediate registers to use LMUL=8,
        // which is the maximum supported grouping for highest throughput.
        vl = __riscv_vsetvl_e16m4(n);

        // Load 16-bit input vectors
        vint16m4_t va = __riscv_vle16_v_i16m4(a, vl);
        vint16m4_t vb = __riscv_vle16_v_i16m4(b, vl);

        // 1. Widening Convert: Promote 'a' to 32-bit (m8) to prevent 
        // intermediate overflow during the accumulation step.
        vint32m8_t v_acc = __riscv_vwcvt_x_x_v_i32m8(va, vl);

        // 2. Fused Widening Multiply-Accumulate: v_acc = (alpha * vb) + v_acc
        // This performs the 16-bit multiplication and 32-bit addition in one 
        // instruction, staying in 32-bit precision as per the scalar ref.
        v_acc = __riscv_vwmacc_vx_i32m8(v_acc, alpha, vb, vl);

        // 3. Narrow and Saturate:
        // vnclip handles the saturation to the signed 16-bit range.
        // Shift 0 is used to match the integer-based 'sat_q15_scalar'.
        vint16m4_t v_res = __riscv_vnclip_wx_i16m4(v_acc, 0, __RISCV_VXRM_RNU, vl);

        // Store result and update pointers
        __riscv_vse16_v_i16m4(y, v_res, vl);

        a += vl;
        b += vl;
        y += vl;
    }
#endif
}

// -------------------- Verification & tiny benchmark -----------------------
static int verify_equal(const int16_t *ref, const int16_t *test, int n, int32_t *max_diff)
{
    int ok = 1;
    int32_t md = 0;
    for (int i = 0; i < n; ++i)
    {
        int32_t d = (int32_t)ref[i] - (int32_t)test[i];
        if (d < 0)
            d = -d;
        if (d > md)
            md = d;
        if (d != 0)
            ok = 0;
    }
    *max_diff = md;
    return ok;
}

#if defined(__riscv)
static inline uint64_t rdcycle(void)
{
    uint64_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}
#endif

int main(int argc, char *argv[])
{

    int ok = 1;
    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    int16_t *a = (int16_t *)aligned_alloc(64, N * sizeof(int16_t));
    int16_t *b = (int16_t *)aligned_alloc(64, N * sizeof(int16_t));
    int16_t *y0 = (int16_t *)aligned_alloc(64, N * sizeof(int16_t));
    int16_t *y1 = (int16_t *)aligned_alloc(64, N * sizeof(int16_t));

    // Deterministic integer data (no libm)
    srand(1234);
    for (int i = 0; i < N; ++i)
    {
        a[i] = (int16_t)((rand() % 65536) - 32768);
        b[i] = (int16_t)((rand() % 65536) - 32768);
    }

    const int16_t alpha = 3; // example scalar gain

    uint32_t c0 = rdcycle();
    q15_axpy_ref(a, b, y0, N, alpha);
    uint32_t c1 = rdcycle();
    printf("Cycles ref: %u\n", c1 - c0);

    int32_t md = 0;

#if defined(__riscv)
    c0 = rdcycle();
    q15_axpy_rvv(a, b, y1, N, alpha);
    c1 = rdcycle();
    ok = verify_equal(y0, y1, N, &md);
    printf("Verify RVV: %s (max diff = %d)\n", ok ? "OK" : "FAIL", md);
    printf("Cycles RVV: %llu\n", (unsigned long long)(c1 - c0));
#endif

    free(a);
    free(b);
    free(y0);
    free(y1);
    return ok ? 0 : 1;
}
