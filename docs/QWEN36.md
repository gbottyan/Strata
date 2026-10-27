# Qwen3.6-35B-A3B: pictures, several requests at once and speed

This page is for people who run [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (35B mixture of
experts, 3B active, 256 experts, top-8) with Strata and want to know how pictures, parallel requests and the prompt
chunk behave, and how far a small card goes at a long context. For sizes and memory, see [MODELS.md](MODELS.md).

Strata runs [Unsloth's GGUFs with the MTP layer](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)
(UD-Q4_K_XL, UD-Q6_K_XL, Q8_0) on its own engine path (`src/qwen35`; `strata` switches to it from the file's
architecture). It runs on NVIDIA and AMD cards. The Intel Arc and Mac engines do not run it.

## Quick start

Pick it in the app's model list, in `strata launch` or in setup:

```sh
strata launch --family qwen36 --model ud-q4_k_xl
strata launch --family qwen36 --model ud-q4_k_xl --vision on --parallel 2
```

| Option | Effect |
| --- | --- |
| `--vision on\|off\|gpu\|cpu` | Terminal: pictures and which reader. Source setup uses `--vision yes\|no\|gpu\|cpu` and `--vision-tokens N`. |
| `--parallel N` | How many requests are answered at once, 1 to 8. Source setup has the same option. |
| `--prefill 512` | Engine flag in the config's `args`, not a `strata launch` option: the prompt chunk size ([Prompt chunk](#prompt-chunk)). |
| `--kv int8\|k8v4` | Engine flag in `args`: the cache's size, 1,056 / 816 bytes a cell ([The cache's size](#the-caches-size-kv-int8-and-kv-k8v4)). |
| `--kv-resident N` | Engine flag in `args`: keep N cells in VRAM and the rest of the cache in pinned host RAM ([The cache's tail](#the-cache-s-tail-kv-resident-n)). |
| `--conversation-cache-mib N` | Engine flag in `args`: park conversations, with their KV, in host RAM ([Parking a conversation](#parking-a-conversation-conversation-cache-mib-n)). |

## Pictures

The vision encoder is Unsloth's `mmproj-F16.gguf` (0.9 GB, pinned revision, SHA-256 checked by setup): a 27-layer ViT
with a 2x2 merger, the same projector family as the Flash-Next models' (`qwen3vl_merger`; only its output width
differs: 2048 here, 2560 there). Strata runs it with the same helper (`strata-vision`, llama.cpp's `mtmd`), so the
encoder choices are the Flash-Next ones ([Images](DETAILS.md#images-vision)).

Pictures are on by default, by the same rule as for every model on NVIDIA and AMD cards:

- The CPU reader runs under 24 GB of graphics memory. It takes a few seconds per picture and no VRAM.
- The card's reader runs from 24 GB (GPU encoder, about 1.2 GB of VRAM at up to 1,024 image tokens). On AMD it is
  `strata-vision-hip`, with the CPU reader as its fallback.

Your explicit choice is kept, with a note. A picture becomes up to 1,024 tokens of the context (the CPU default is 300).
The engine puts the encoder's rows where the prompt has `<|image_pad|>` tokens and gives every row its 2-D position,
interleaved M-RoPE as llama.cpp does for this model (rotary sections 11 / 11 / 10).

Pictures work with several requests at once, with the MTP drafts and with the conversation cache. The same conversation
sent again continues from the held state when its pictures are the same (matched by the server's key for each
picture). A different picture starts that conversation again.

> **Measured on:** RTX 3060 12 GB (CUDA) and an R9700 box (32 threads) and a 3060 box (10 threads), UD-Q4_K_XL, Strata
> 0.1.45.

| Encoder | Time per picture (640x400 = 260 tokens / 800x450 = 350 tokens) | VRAM |
| --- | --- | --- |
| GPU, RTX 3060 | 0.10 s / 0.15 s | 1,179 MiB at up to 1,024 tokens (1,337 MiB at 2,048), CUDA context included |
| CPU, 32 threads (R9700 box) | 0.9 s / 1.1 s | none |
| CPU, 10 threads (3060 box) | 1.6 s / 2.1 s | none |

## Several requests at once

`"parallel": N` in the model's config (`strata-qwen36-<size>.json`) or `--parallel N` makes the engine keep N
conversations open (`--batch N`, up to 8) and decode them together. Every window carries one token of each
conversation, so the dense weights, the head and every routed expert that conversations share are read once for all of
them. The server speaks the same engine protocol as for Flash-Next ([BATCHING.md](BATCHING.md)).

A new request waits only for the read of its own prompt, which runs between the decode windows of the others. A
conversation continues in the slot that holds its start, or the engine copies the longer held start from another slot.
A request that runs alone (the solo path, with its MTP drafts) moves into a batch slot when another arrives, and back
when it is alone again, and gives the same text either way.
`STRATA_Q36_READ_WINDOWS=4` lets the decoding slots produce more tokens during a long prompt read, at the cost of a
slower read. The default is one decode window per chunk.

**MTP drafts.** A request that has the engine to itself keeps its MTP drafts. With several requests the batch window has
no drafts unless `--batch-mtp` is in the config's `args` (opt-in). Each slot's window then also carries the proposal its
MTP block made, one per slot, up to the rows to spare in the 8-row window.

**What a slot costs.** Each slot keeps its own cache in VRAM: 1.5 GB at a context of 32K (fp32 keys and values of the 10
attention layers and the MTP block, 45 KB per token) and 0.07 GB of recurrent state. With `--kv int8` / `--kv k8v4` the
same cache is 14.3 / 11.6 KB a token, 0.45 / 0.36 GB at 32K. Setup writes the context for one conversation, so N slots
take N times that from the room the experts would have. On a card that does not hold every
expert (12 and 16 GB, the hybrid mode) the expert slots lose that room, and a batch reads more distinct experts from RAM.

Setup offers `--parallel` where the sessions do not eat the experts' room, says the price in the same breath and asks
"How many requests at once?" Enter keeps one at a time, and any number you give is kept.

| Card type | Slots setup offers |
| --- | --- |
| Every expert still fits | Up to 3 to 4 (3 on a 32 GB card at 32K) |
| Hybrid mode | The count whose sessions take at most 35% of the room the experts would have. 12 GB: 4 slots at 8K, none at 32K. 16 GB: 2 slots at 32K. |

The engine says what the sessions take before it allocates them and refuses, with the numbers, a count that cannot fit.
The table is the fp16 cache's arithmetic; a quantised cache takes a quarter to a third of that room, and setup's offers
have not been re-measured against it.

In the hybrid mode the text of a request can differ a little between running alone and running beside others. The
experts that sit on the GPU or on the CPU change with the routing of all the requests, and the two compute with
different rounding. With the slot tables fixed (`STRATA_Q36_FIXED_SLOTS=1`, a test setting) every slot equals its solo
run there too.

## Prompt chunk

The MoE prompt kernels read every expert once per chunk, so a bigger chunk feeds them more rows per expert. With every
expert in VRAM the engine takes the largest of 4096 / 2048 / 1024 that fits beside the experts, the 0.77 GiB reserve
and the sessions' caches.

| Situation | Chunk |
| --- | --- |
| Every expert in VRAM, 8K to 64K context | 4096 |
| Every expert in VRAM, 100K context | 2048 |
| Every expert in VRAM, 131K context | 512 |
| `--batch 2..4` | 1024 at most, so the other slots are not held for a whole chunk |
| `--batch 8` | 512 |
| Hybrid mode | 2048; 1024 or 512 on a card with 7.5 or 5.5 GiB free |

In the hybrid mode a bigger chunk takes expert slots (an RTX 3060 at chunk 4096 read prompts 5% faster and decoded 13%
slower). `--prefill 512` (or `STRATA_Q36_CHUNK=512`) gives the 512 size, and then the text is the base engine's, bit
for bit. The chunk size changes the GEMM tiles the BLAS picks and with them the last bits of the prompt state. This is
the size of the llama.cpp CPU-against-HIP difference on the same file, not a quality loss. A prompt shorter than the
chunk (every chat turn after the first) is one chunk either way.

## Speed and exactness

Every change on the decode path gives the old kernels' bits. `STRATA_Q36_VERIFY=1 strata ... --no-graph` runs the old
kernel beside each new one on the real activations and compares them bit for bit. `STRATA_Q36_OLD_KERNELS=1` switches
the new kernels off. Gate/up for Q4_K experts uses the V2K kernels (`STRATA_EXPERT_V2K`, the default on gfx12).

That claim is about the fp16 cache. `--kv int8` and `--kv k8v4` quantise the cache, so they are not bit-exact by
design: `tools/q36_attn_test.cu` builds a cache the way the engine writes one and compares the attention accumulation
against a CPU reference, 5.3e-7 relative for int8 at every group size and 6.7e-7 for k8v4. The text of a long
conversation will differ from the fp16 cache's somewhere.

> **Measured on:** AMD Radeon AI PRO R9700 32 GB (HIP gfx1201), UD-Q4_K_XL, Strata 0.1.45 against the engine before it,
> 5 interleaved pairs, medians; llama.cpp 972b5fc HIP on the same file (`-fa 1`).

| | Before | 0.1.45 | llama.cpp |
| --- | ---: | ---: | ---: |
| Decode tg128 | 83.2 tok/s | 117.8 | 81.9 to 82.1 |
| Prompt pp4096 | 3,453 | 6,220 (chunk 4096) | 3,794 (`-ub 2048`) / 3,042 (`-ub 512`) |

More numbers are in [History](#history--older-measurements).

## Long context on a small card

An 8 GB card runs this model in the hybrid mode: 4.40 GiB of weights in VRAM, all 18.75 GiB of routed experts in pinned
host RAM. What is left is the context, so the size of the KV cache is most of the game. Four things changed it.

> **Measured on:** GTX 1070 Ti 8 GB (CUDA sm_61, ~256 GB/s), UD-Q4_K_XL, one slot, `--mtp 2`, chunk 512.

### The cache's size: `--kv int8` and `--kv k8v4`

`--kv k8v4` keeps each cell as INT8 keys plus a Hadamard-rotated Q4_0 value: **816 bytes a cell for both heads against
int8's 1,056**, 11,616 bytes a token against int8's ~14,237 (the engine prints the number at boot). A 128K context is
1.42 GiB of KV where int8 needs 1.74.

The value's 4-bit block stores its codes only: the fp16 scale lives in a scale table beside the cache, which is where
both readers - the attention kernel, and the fp16 mirror that feeds the prompt path's PV GEMM - go to read it. That
makes a head's value row 8 x 16 = 128 bytes and puts every block on a 16-byte boundary, so a thread's byte is one of
four in an aligned word and a warp's 32 lanes cover one block's codes inside a single sector.

### The cache's tail: `--kv-resident N`

`--kv-resident N` keeps N cells of each attention layer in VRAM and the rest of the cache in pinned host RAM, which the
attention kernel reads a cell at a time; the freed VRAM goes to the expert staging the engine wants. At 131072 cells
with N = 32768:

    k8v4 KV: 32768 cells resident in VRAM, 98304 spilled to 0.73 GiB of pinned host RAM (1 sessions, 0.73 GiB of VRAM freed)

and 0.61 GiB free becomes 1.36. The minimum, 20480, frees 0.86 GiB there.

**What is not measured is the spill's cost at the depth it is for.** The runs that checked it decoded a 4,000-token
prompt (26 tok/s), where every cell being read is resident. The penalty at full depth is unconfirmed.

### Parking a conversation: `--conversation-cache-mib N`

q36 keeps a conversation's state in VRAM, so without this a second conversation wipes the first and reuse is capped at
one conversation. The cache parks a conversation's KV and its fixed-size recurrent state in host RAM, keyed by token
ids, least recently used out, and says what it holds:

    conversation cache: 4096 MiB of host RAM, up to 8 conversations. A conversation's running state is 0.06 GiB
    (fixed: the GDN state is not per token) and its KV 11616 B/token, so one at 131072 tokens is 1.48 GiB and the
    cache holds about 2 of them; parking evicts the least recently used.

### Attention: the query heads share a block

The attention layers have 8 query heads to every key-value head, and reading K and V once per group of query heads
instead of once per head is the biggest single win on this card. `STRATA_Q36_GQ` sets the group (2, 4, 8; default 8).
At 48K, GQ 2 / 4 / 8: attention 15.13 / 12.09 / 11.40 ms a token, decode 19.4 / 20.7 / 20.8 tok/s.

### What it adds up to

> Context 98304, pp96000, tg60, one repetition each. Attention is the profiler's per-token line, two windows, mean.

| | attention ms/token | decode tg60 | prompt pp96000 |
| --- | ---: | ---: | ---: |
| int8, GQ 2 (before) | 44.7 | 12.2 tok/s | 164.0 |
| int8, GQ 2, shared blocks | 28.2 | 15.4 | - |
| int8, GQ 8 | 20.7 | 16.5 | 163.6 |
| k8v4, GQ 8 | 20.3 | 17.6 | 163.5 |
| k8v4, GQ 8, codes-only row | **19.4** | **17.9** | 163.5 |

Prefill did not move: 163.5 tok/s in every mode. And the attention kernel is not reading-bound - at 816 bytes a cell a
token reads 246 MB of value, 1 ms at 256 GB/s against the 19 measured - so what is left is the loop's own arithmetic
and the occupancy its 17 KiB of shared memory allows (ptxas: 64 registers, 17,440 bytes, three blocks per SM).

`STRATA_Q36_PROFILE=1 strata ...` prints the stages per token: `attn 19.08, GDN 7.23, experts 25.92, embed 0.01, head
2.98, MTP draft 0.00 ms/token = 55.87`. The `bench N:` line is ground truth; the windows are one-repetition noisy.

### What fits in 6 GB

The floor is 5.97 GiB at a context of 2,048: 4.40 GiB of weights, and ~1.6 GiB of engine buffers, the MTP draft layer
and the CUDA context. `--chunk 128` frees 0.18 GiB; `--mtp 0` does not unload the draft layer. A 6 GB card therefore
fits about **16K of context**, not the 128K this box runs. The build is `CMAKE_CUDA_ARCHITECTURES=61` and embeds sm_61
PTX, so a Pascal card runs it natively and a Turing card JITs the PTX. The bigger constraint is system RAM: 18.75 GiB
of pinned experts, plus the parked conversations and any spilled KV.

## Limits

- Eight slots at most (the window holds 8 rows). `--batch-mtp` shares those rows between slots and proposals.
- A sampling request in a batch decodes one token per window without proposals.
- `logit_bias` is not available on this engine path. The server refuses a request that has it (as it does with
  `--batch` for Flash-Next).
- A slot that stopped being alone goes back to the solo path only through the server's own hand-over. The MTP block's
  cache has holes for the tokens a slot decoded in batch windows, so the drafts of a later solo stretch are a little
  worse. The text is not affected.
- Intel Arc (SYCL) and Mac do not run this family.

## Testing

The gates are in `tools/qwen35/` and talk to the engine through its serve protocol (no server needed). Run them under
the box's GPU lock.

| Script | Checks |
| --- | --- |
| `q36_solo_gate.py` + `q36_cmp_gate.py` | The 10 parity prompts x 128 greedy tokens through `GEN`, compared with another build's. |
| `q36_batch_gate.py --slots N --waves --long 2500` | Every slot's tokens equal its solo tokens, slots reused, a long prompt read beside decoding slots. |
| `q36_proto_edge.py` | `max_new` 1, `BSTOP`, a busy slot, `STOP` during a prompt read, held-state reuse, sampled requests, the context's end. |
| `q36_interleave_probe.py` | What the decoding slots see while a long prompt is read. |
| `q36_vision_cmp.py` (with `q36_mtmd_dump.cpp` built against llama.cpp) | The picture comparison against llama.cpp. |
| `q36_vision_proto.py` | Pictures through `GENI` / `BGENI` (solo, in a batch, the conversation cache). |
| `q36_vision_serve.py`, `q36_serve_par.py`, `q36_serve_handoff.py` | The same through the server. |
| `q36_par_bench.py` | The aggregate-throughput A/B against `llama-server -np N`. |

## History / older measurements

### Pictures checked against llama.cpp

> **Measured on:** R9700 32 GB (HIP), UD-Q4_K_XL, Strata 0.1.45. llama.cpp's `mtmd` and its own graph, the encoder on the
> CPU on both sides, the same prompt ids, llama.cpp's answer fed to both, logits compared step by step over 24 steps.
> The floor is llama.cpp's CPU backend against its own HIP run.

| Picture and question | Prompt tokens | First token | Mean KL, llama.cpp HIP vs Strata | Floor |
| --- | ---: | --- | ---: | ---: |
| 640x400, text in the image | 288 | same | 0.029 | 0.056 |
| 640x400, list the shapes | 294 | same | 0.035 | 0.039 |
| 800x450, text in the image | 378 | same | 0.021 | 0.031 |
| 800x450, describe the shapes | 381 | same | 0.0005 | 0.0019 |
| 1024x1024, the title (1,024 image tokens, 3 chunks) | 1,042 | same | 0.0022 | 0.0026 |
| two pictures in one request | 655 | differs (a near tie; the floor's KL at this step is 0.16) | 0.014 | 0.022 |

The answers are the same text as llama.cpp's. With the picture options on, a request without pictures gives the same
tokens as without them (10 prompts x 128 greedy tokens).

### Aggregate decode speed, several requests at once

> **Measured on:** same prompts, greedy, 128 new tokens each, wall time from first request to last answer, 5
> interleaved pairs, medians, Strata 0.1.45 with `--batch 8 --mtp 0` against `llama-server -np N`.

| Card | Requests | Strata tok/s | llama.cpp | Ratio |
| --- | ---: | ---: | ---: | ---: |
| R9700 32 GB, UD-Q4_K_XL | 1 | 83.9 | 73.8 | 1.14 |
| | 2 | 173.5 | 107.4 | 1.61 |
| | 4 | 252.0 | 165.0 | 1.53 |
| | 8 | 325.4 | 208.1 | 1.56 |
| RTX 3060 12 GB, hybrid (llama.cpp `-ncmoe 28 -t 8`, the best split of a sweep) | 1 | 64.0 | 41.1 | 1.56 |
| | 2 | 85.1 | 52.5 | 1.62 |
| | 4 | 99.3 | 63.0 | 1.58 |

All pairs were faster for Strata. Every slot's tokens equal the tokens of the same request alone.

### Several requests with the base engine, and with `--batch-mtp`

> **Measured on:** R9700 32 GB, 5 interleaved pairs, Strata 0.1.45 (before the speed work below).

| Requests | `--mtp 0` | `--mtp 2 --batch-mtp` | Ratio |
| ---: | ---: | ---: | ---: |
| 1 | 83.7 tok/s | 134.0 | 1.60 |
| 2 | 173.1 | 207.0 | 1.20 |
| 4 | 251.0 | 275.0 | 1.10 |
| 8 | 324.0 | 320.5 | 0.99 (no row to spare: no proposals) |

> **Measured on:** hybrid cards, UD-Q4_K_XL, medians of 5 interleaved pairs, `--mtp 2`; "together" is the total tokens
> per second of N simultaneous requests against one request at a time with the draft layer.

| Card, context | Slots | One request alone, against 1 slot | N together, against one at a time |
| --- | ---: | ---: | ---: |
| RTX 3060 12 GB, 8K | 2 / 4 / 8 | -3.5% / -4.3% / -12% | 0.99x / 1.15x / 1.10x |
| RTX A4000 16 GB, 32K | 2 / 4 | -10.6% / -29.9% | 1.03x / 0.76x |
| Tesla P100 16 GB, 32K | 2 / 4 | -7% / -23% | 0.98x / 0.81x |

Without the draft layer (`--mtp 0`) the same slots give 1.26x to 1.64x together (3060 at 8K: 4 slots 1.64x for -5.5%
alone). A single request with drafts is already about 1.6x as fast as one without, so on a small card the main gain is
that a second user starts at once. On a card that holds every expert (R9700 32 GB, 8K prompts), 4 slots run 1.45x
one-at-a-time-with-drafts (1.59x with `--batch-mtp`).

### Prompt read beside decoding slots

> **Measured on:** R9700 32 GB, Strata 0.1.45, three decoding slots.

A 3,008-token prompt was read in 0.95 s (3,200 tok/s, 6 chunks). Each decoding slot paused at most 160 ms (one chunk)
and produced 20 tokens meanwhile. With `STRATA_Q36_READ_WINDOWS=4` they produced 65 tokens during a 1.14 s read.

### Speed work in 0.1.45

> **Measured on:** R9700 32 GB (HIP gfx1201), UD-Q4_K_XL, 5 interleaved pairs, 3 repetitions each, medians; base = the
> engine before 0.1.45.

| | Base | 0.1.45 | Ratio |
| --- | ---: | ---: | ---: |
| `--mtp 2`, 10 prompts greedy 128 | 167.7 tok/s | 183.7 | 1.10 |
| 1 / 2 / 4 / 8 requests (`--batch 8 --mtp 0`) | 85.2 / 182.4 / 271.9 / 361.1 | 122.4 / 205.1 / 303.3 / 386.9 | 1.44 / 1.12 / 1.12 / 1.07 |
| Prompt pp4096 at chunk 512 / 1024 / 2048 / 4096 | | 3,453 / 4,786 / 5,766 / 6,223 | |

UD-Q6_K_XL on the same card (every main layer's experts in VRAM, chunk stays 512): decode 80.4 to 106.9 tok/s, pp4096
2,160 to 2,200 (llama.cpp 3,486 at `-ub 2048`, 2,738 at `-ub 512`), `--mtp 2` 154.7 to 167.3. `--chunk 2048` there moves
the experts to the hybrid placement (205 of 256 per layer in VRAM): prompt 3,500 tok/s, decode 94. That is an option,
not the default.

> **Measured on:** RTX 3060 12 GB (CUDA sm_86, 20 threads), UD-Q4_K_XL in the hybrid mode (37 of 256 experts per layer
> in VRAM), Strata 0.1.45.

Decode tg128 58.5 to 59.7 tok/s (+2%), pp2048 1,958 to 2,022 (+3.3%), 10/10 prompts identical, plain and with `--mtp 2`.
llama.cpp 972b5fc CUDA (`-ncmoe 26 -t 8 -ub 2048`): pp2048 1,328, tg128 48 to 52. The GPU part of a token is the dense
Q8_0 at the card's bandwidth and the rest waits for the CPU pool, so only the small kernels' share is gained.

KL against the 512 chunk on six real 1.5K to 3.2K-token documents (48 teacher-forced rows each): chunk 1024 4.0e-3,
2048 4.2e-3, 4096 3.6e-3, top-1 agreement 44 to 48 of 48 rows. Greedy 64 tokens after those documents: the default chunk
gives the base engine's text for 2 of 6, `--prefill 512` for 6 of 6.
