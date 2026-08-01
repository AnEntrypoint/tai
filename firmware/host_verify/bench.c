#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../common/llm.h"

static uint8_t *read_file(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *b = malloc(*n);
  if (fread(b, 1, *n, f) != *n) { fprintf(stderr, "short read\n"); exit(1); }
  fclose(f); return b;
}

static double now_s(void) {
  return (double)clock() / CLOCKS_PER_SEC;
}

int main(int argc, char **argv) {
  const char *bin = argc > 1 ? argv[1] : "firmware/model/model.bin";
  int n_gen = argc > 2 ? atoi(argv[2]) : 200;
  size_t n;
  uint8_t *buf = read_file(bin, &n);
  Model m;
  if (llm_load(buf, &m)) { fprintf(stderr, "bad magic\n"); return 1; }

  int D = m.c.dim, L = m.c.n_layers, P = m.c.ple_dim, F = m.c.ffn, V = m.c.vocab, S = m.c.seq_len;
  Scratch s;
  s.x = malloc(D * 4); s.h = malloc((F > D ? F : D) * 4);
  s.qkv = malloc(3 * D * 4); s.att = malloc(D * 4);
  s.g1 = malloc(F * 4); s.g2 = malloc((P > F ? P : F) * 4);
  s.ple = malloc(L * P * 4); s.tmpP = malloc(L * P * 4); s.trow = malloc(L * P * 4);
  s.logits = malloc(V * 4);
  s.scores = malloc(S * 4);
  s.kcache = malloc((size_t)L * S * D * 4);
  s.vcache = malloc((size_t)L * S * D * 4);

  int prompt[] = {1, 500, 1000, 200, 42, 777, 13, 99};
  int plen = sizeof(prompt) / sizeof(int);
  for (int i = 0; i < plen; i++) llm_forward(&m, prompt[i], i, &s);
  int pos = plen, tok = 0;

  double t0 = now_s();
  int decoded = 0;
  for (int step = 0; step < n_gen && pos < S; step++) {
    int best = 0; float bv = -1e30f;
    for (int v = 0; v < V; v++)
      if (s.logits[v] > bv) { bv = s.logits[v]; best = v; }
    tok = best;
    llm_forward(&m, tok, pos++, &s);
    decoded++;
  }
  double dt = now_s() - t0;
  printf("C host: %d tokens in %.2f s   %.2f tok/s (%.1f ms/token)\n",
         decoded, dt, decoded / dt, dt * 1000.0 / decoded);
  return 0;
}
