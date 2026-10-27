// detnum_hotpatch: in-process, verified hot patch that routes RCPPS/RSQRTPS sites of a shared library
// through a table-driven bit-exact emulator (rcp_emu.h). Nothing on disk is modified.
// Each patch window becomes one trampoline made of ops: EMU (emulated approx instruction) or RAW
// (relocated original instruction, RIP-relative disp32 rewritten). Windows never overlap.
// Built on demand by dex_benchmark.sim.determinism (cc -O2 -fPIC -shared -mgeneral-regs-only).
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "rcp_emu.h"

#define MAX_COUNTERS 512
#define MAX_OPS 6
typedef struct {
  uint64_t off;                 // link-time vaddr of the window start (an approx instruction)
  uint32_t patch_len;           // bytes overwritten (>= 5)
  uint32_t nops;
  uint8_t orig[48];             // original window bytes (verified before patching)
  uint32_t op_type[MAX_OPS];    // 1 = EMU, 0 = RAW
  uint32_t op_kind[MAX_OPS];    // EMU: 0 rcpps, 1 rsqrtps
  uint32_t op_dst[MAX_OPS], op_src[MAX_OPS];
  uint32_t op_counter[MAX_OPS]; // EMU: counter index (relative to index_base)
  uint64_t op_off[MAX_OPS];     // RAW: original vaddr
  uint32_t op_len[MAX_OPS];     // RAW: length
  uint32_t op_rip[MAX_OPS];     // RAW: 1 if ends with RIP-relative disp32
  uint8_t op_bytes[MAX_OPS][16];
} window_t;

static uint8_t *g_tab_buf;
static long g_tab_len;
static rcp_tables g_tab;
static uint64_t g_hits[MAX_COUNTERS];

__attribute__((used)) void detnum_emu(uint32_t *lanes, uint32_t kind_ctr) {
  uint32_t kind = kind_ctr & 0xff, c = kind_ctr >> 8;
  if (c < MAX_COUNTERS) __atomic_fetch_add(&g_hits[c], 1, __ATOMIC_RELAXED);
  for (int i = 0; i < 4; i++) lanes[i] = kind ? emu_rsqrt(&g_tab, lanes[i]) : emu_rcp(&g_tab, lanes[i]);
}

uint64_t detnum_hits(int i) { return (i >= 0 && i < MAX_COUNTERS) ? g_hits[i] : 0; }

static uint64_t find_base(const char *name_suffix, char *err) {
  FILE *f = fopen("/proc/self/maps", "r"); if (!f) { strcpy(err, "maps"); return 0; }
  char line[1024]; uint64_t base = 0;
  while (fgets(line, sizeof line, f)) {
    unsigned long lo, hi, off; char perms[8], path[700] = {0};
    if (sscanf(line, "%lx-%lx %7s %lx %*s %*s %699s", &lo, &hi, perms, &off, path) >= 4) {
      size_t n = strlen(path), m = strlen(name_suffix);
      if (n >= m && strcmp(path + n - m, name_suffix) == 0 && off == 0) { base = lo; break; }
    }
  }
  fclose(f);
  if (!base) strcpy(err, "library not mapped");
  return base;
}

static uint8_t *emit(uint8_t *p, const void *b, size_t n) { memcpy(p, b, n); return p + n; }

static uint8_t *alloc_near(uint64_t lo, uint64_t hi, size_t size) {
  long pg = sysconf(_SC_PAGESIZE);
  for (int64_t d = 16 << 20; d < (int64_t)1800 << 20; d += 16 << 20) {
    uint64_t cands[2] = {(lo > (uint64_t)d ? lo - d : 0) & ~(uint64_t)(pg - 1), (hi + d) & ~(uint64_t)(pg - 1)};
    for (int c = 0; c < 2; c++) {
      if (!cands[c]) continue;
      void *p = mmap((void *)cands[c], size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
      if (p != MAP_FAILED) return p;
    }
  }
  return NULL;
}

static uint8_t *emit_emu(uint8_t *p, uint32_t kind, uint32_t dst, uint32_t src, uint32_t ctr) {
  // lea rsp,[rsp-128]; pushfq; push rax,rcx,rdx,rsi,rdi,r8,r9,r10,r11,rbp; mov rbp,rsp; and rsp,-16; sub rsp,16
  static const uint8_t pre[] = {0x48, 0x8D, 0x64, 0x24, 0x80, 0x9C, 0x50, 0x51, 0x52, 0x56, 0x57, 0x41, 0x50, 0x41, 0x51, 0x41, 0x52, 0x41, 0x53,
                                0x55, 0x48, 0x89, 0xE5, 0x48, 0x83, 0xE4, 0xF0, 0x48, 0x83, 0xEC, 0x10};
  p = emit(p, pre, sizeof pre);
  if (src >= 8) *p++ = 0x44;                                  // movups [rsp], xmmS
  *p++ = 0x0F; *p++ = 0x11; *p++ = (uint8_t)(((src & 7) << 3) | 4); *p++ = 0x24;
  static const uint8_t movrdi[] = {0x48, 0x89, 0xE7}; p = emit(p, movrdi, 3);   // mov rdi,rsp
  *p++ = 0xBE; uint32_t ks = kind | (ctr << 8); p = emit(p, &ks, 4);            // mov esi, kind|ctr<<8
  *p++ = 0x48; *p++ = 0xB8; uint64_t fn = (uint64_t)&detnum_emu; p = emit(p, &fn, 8);
  *p++ = 0xFF; *p++ = 0xD0;                                   // call rax
  if (dst >= 8) *p++ = 0x44;                                  // movups xmmD, [rsp]
  *p++ = 0x0F; *p++ = 0x10; *p++ = (uint8_t)(((dst & 7) << 3) | 4); *p++ = 0x24;
  // mov rsp,rbp; pop rbp,r11,r10,r9,r8,rdi,rsi,rdx,rcx,rax; popfq; lea rsp,[rsp+128]
  static const uint8_t post[] = {0x48, 0x89, 0xEC, 0x5D, 0x41, 0x5B, 0x41, 0x5A, 0x41, 0x59, 0x41, 0x58, 0x5F, 0x5E, 0x5A, 0x59, 0x58,
                                 0x9D, 0x48, 0x8D, 0xA4, 0x24, 0x80, 0x00, 0x00, 0x00};
  return emit(p, post, sizeof post);
}

#define TSLOT 512
int detnum_patch(const char *lib_suffix, const uint8_t *tables, long tlen, const window_t *w, int n, int index_base, char *err) {
  if (!g_tab_buf) {
    g_tab_buf = malloc(tlen); memcpy(g_tab_buf, tables, tlen); g_tab_len = tlen;
    if (load_tables(g_tab_buf, tlen, &g_tab)) { strcpy(err, "bad tables"); return -1; }
  } else if (tlen != g_tab_len || memcmp(g_tab_buf, tables, tlen)) { strcpy(err, "tables differ from first install"); return -1; }
  uint64_t base = find_base(lib_suffix, err); if (!base) return -1;
  for (int i = 0; i < n; i++) {
    if (w[i].patch_len < 5 || w[i].patch_len > sizeof w[i].orig || w[i].nops > MAX_OPS) { sprintf(err, "bad window %d", i); return -2; }
    if (memcmp((void *)(base + w[i].off), w[i].orig, w[i].patch_len)) { sprintf(err, "original bytes differ at window %d off 0x%lx", i, (unsigned long)w[i].off); return -2; }
    for (int j = 0; j < n; j++)
      if (j != i && w[j].off < w[i].off + w[i].patch_len && w[i].off < w[j].off + w[j].patch_len) { sprintf(err, "overlapping windows %d %d", i, j); return -2; }
    for (uint32_t o = 0; o < w[i].nops; o++)
      if (w[i].op_type[o] == 1 && index_base + w[i].op_counter[o] >= MAX_COUNTERS) { strcpy(err, "counter overflow"); return -2; }
  }
  uint64_t lo = base + w[0].off, hi = lo;
  for (int i = 0; i < n; i++) { uint64_t a = base + w[i].off; if (a < lo) lo = a; if (a > hi) hi = a; }
  size_t tsz = (size_t)TSLOT * n; tsz = (tsz + 4095) & ~(size_t)4095;
  uint8_t *tr = alloc_near(lo, hi, tsz); if (!tr) { strcpy(err, "no trampoline memory within 2GB"); return -3; }
  for (int i = 0; i < n; i++) {
    const window_t *s = &w[i]; uint8_t *p = tr + (size_t)TSLOT * i, *start = p; uint64_t site = base + s->off;
    if ((int64_t)((uint64_t)start - site) > 0x7fff0000LL || (int64_t)(site - (uint64_t)start) > 0x7fff0000LL) { strcpy(err, "trampoline out of range"); return -3; }
    for (uint32_t o = 0; o < s->nops; o++) {
      if (s->op_type[o] == 1) { p = emit_emu(p, s->op_kind[o], s->op_dst[o], s->op_src[o], index_base + s->op_counter[o]); continue; }
      uint8_t *ins = p; p = emit(p, s->op_bytes[o], s->op_len[o]);
      if (s->op_rip[o]) {
        int32_t disp; memcpy(&disp, ins + s->op_len[o] - 4, 4);
        uint64_t target = base + s->op_off[o] + s->op_len[o] + (int64_t)disp;
        int64_t nd = (int64_t)target - (int64_t)((uint64_t)ins + s->op_len[o]);
        if (nd > 0x7fffffffLL || nd < -0x80000000LL) { strcpy(err, "rip reloc out of range"); return -4; }
        int32_t nd32 = (int32_t)nd; memcpy(ins + s->op_len[o] - 4, &nd32, 4);
      }
    }
    uint64_t resume = site + s->patch_len;
    *p++ = 0xE9; int32_t rel = (int32_t)((int64_t)resume - (int64_t)((uint64_t)p + 4)); p = emit(p, &rel, 4);
    if (p - start > TSLOT) { strcpy(err, "trampoline overflow"); return -5; }
  }
  if (mprotect(tr, tsz, PROT_READ | PROT_EXEC)) { strcpy(err, "mprotect tramp"); return -6; }
  long pg = sysconf(_SC_PAGESIZE);
  for (int i = 0; i < n; i++) {
    const window_t *s = &w[i]; uint64_t site = base + s->off; uint8_t *page = (uint8_t *)(site & ~(uint64_t)(pg - 1));
    size_t span = (size_t)(((site + s->patch_len - 1) & ~(uint64_t)(pg - 1)) - (uint64_t)page) + pg;
    if (mprotect(page, span, PROT_READ | PROT_WRITE | PROT_EXEC)) { strcpy(err, "mprotect site rw"); return -7; }
    uint8_t jmp[48]; memset(jmp, 0x90, sizeof jmp); jmp[0] = 0xE9;
    int32_t rel = (int32_t)((int64_t)(uint64_t)(tr + (size_t)TSLOT * i) - (int64_t)(site + 5)); memcpy(jmp + 1, &rel, 4);
    memcpy((void *)site, jmp, s->patch_len);
    if (mprotect(page, span, PROT_READ | PROT_EXEC)) { strcpy(err, "mprotect site rx"); return -8; }
  }
  sprintf(err, "ok base=0x%lx tramp=%p windows=%d", (unsigned long)base, (void *)tr, n);
  return 0;
}
