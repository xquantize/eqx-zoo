---
name: Accelerator verification report
about: Report how eqx-zoo's numerical checks behave on a GPU or TPU
title: "Accelerator report: <model> on <device>"
labels: ["numerics"]
---

Thank you for running eqx-zoo on an accelerator! Comparing against Hugging Face **on the same device** keeps framework differences separate from device arithmetic. Fill in what you can; partial reports are welcome.

**Setup**

- Model and revision:
- eqx-zoo commit:
- Device:
- jax / jaxlib versions:

**Hugging Face float32, on the same device**

- Attention backend:
- TF32 / reduced-precision float32 setting:

**JAX float32**

- Default matmul precision:
- Highest matmul precision:

**JAX bfloat16**

- Eager (op by op):
- Whole-model JIT:

**Inputs**

- Short prompt:
- Long prompt (about 256+ tokens):

**Metrics** (for each of the above)

- RMS error
- Max absolute error
- Top-1 / top-k agreement
- Greedy token agreement

**If generation is relevant**

- Full forward (teacher-forced):
- Cached decoding:

The structure of this template was proposed in a detailed community report on an NVIDIA L4.
