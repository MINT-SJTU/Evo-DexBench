// Table-driven, bit-exact emulation of x86 RCPPS / RSQRTPS (per-vendor tables).
// Integer-only (safe to call from a trampoline compiled with -mgeneral-regs-only).
#pragma once
#include <stdint.h>

typedef struct {
  int k_rcp, k_rs0, k_rs1;           // leading-mantissa-bit counts that determine the output
  const uint32_t *rcp, *rs0, *rs1;   // outputs for inputs in [1,2) (rcp, rsqrt) and [2,4) (rsqrt)
} rcp_tables;

static inline uint32_t emu_rcp(const rcp_tables *t, uint32_t x) {
  uint32_t s = x & 0x80000000u, e = (x >> 23) & 0xff, m = x & 0x7fffffu;
  if (e == 0) return s | 0x7f800000u;                 // +-0 and denormals -> +-inf
  if (e == 255) return m ? (x | 0x400000u) : s;        // NaN -> quiet NaN, inf -> +-0
  uint32_t b = t->rcp[m >> (23 - t->k_rcp)];
  int eb = (int)((b >> 23) & 0xff) - ((int)e - 127);
  if (eb <= 0) return s;                               // result in denormal range -> flushed to +-0
  return s | ((uint32_t)eb << 23) | (b & 0x7fffffu);
}

static inline uint32_t emu_rsqrt(const rcp_tables *t, uint32_t x) {
  uint32_t s = x & 0x80000000u, e = (x >> 23) & 0xff, m = x & 0x7fffffu;
  if (e == 0) return s | 0x7f800000u;                 // +-0 and denormals -> +-inf
  if (e == 255) { if (m) return x | 0x400000u; return s ? 0xffc00000u : 0u; }
  if (s) return 0xffc00000u;                           // negative normal -> default NaN
  int tt = (int)e - 127, k2 = tt >> 1, par = tt & 1;   // arithmetic shift == floor
  uint32_t b = par ? t->rs1[m >> (23 - t->k_rs1)] : t->rs0[m >> (23 - t->k_rs0)];
  int eb = (int)((b >> 23) & 0xff) - k2;
  return ((uint32_t)eb << 23) | (b & 0x7fffffu);
}

// tables.bin layout written by rcp_census: [k][2^k u32] x3 (rcp, rsqrt[1,2), rsqrt[2,4))
static inline int load_tables(const uint8_t *buf, long len, rcp_tables *t) {
  long o = 0; const uint32_t **dst[3] = {&t->rcp, &t->rs0, &t->rs1}; int *kk[3] = {&t->k_rcp, &t->k_rs0, &t->k_rs1};
  for (int i = 0; i < 3; i++) {
    if (o + 4 > len) return -1;
    int k = *(const int *)(buf + o); o += 4;
    if (k < 1 || k > 23 || o + (4L << k) > len) return -1;
    *kk[i] = k; *dst[i] = (const uint32_t *)(buf + o); o += 4L << k;
  }
  return o == len ? 0 : -1;
}
