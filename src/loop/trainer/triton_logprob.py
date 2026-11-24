import time
from typing import Any

import torch
import torch.nn.functional as F

try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _fused_logprob_fwd_kernel(
        logits_ptr,
        targets_ptr,
        out_lp_ptr,
        out_lse_ptr,
        stride_row,
        vocab_size,
        BLOCK_V: tl.constexpr,
    ):
        """Online log-sum-exp pass over vocab V without materializing [N, V] fp32 softmax."""
        row = tl.program_id(0)
        row_ptr = logits_ptr + row * stride_row
        target = tl.load(targets_ptr + row)

        m_i = -float("inf")
        l_i = 0.0
        for col_start in range(0, vocab_size, BLOCK_V):
            cols = col_start + tl.arange(0, BLOCK_V)
            mask = cols < vocab_size
            x = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
            block_max = tl.max(x, axis=0)
            m_new = tl.maximum(m_i, block_max)
            l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
            m_i = m_new

        lse = m_i + tl.log(l_i)
        target_logit = tl.load(row_ptr + target).to(tl.float32)
        tl.store(out_lse_ptr + row, lse)
        tl.store(out_lp_ptr + row, target_logit - lse)

    @triton.jit
    def _fused_logprob_bwd_kernel(
        logits_ptr,
        targets_ptr,
        lse_ptr,
        grad_out_ptr,
        grad_logits_ptr,
        stride_row,
        vocab_size,
        BLOCK_V: tl.constexpr,
    ):
        """Analytical backward: dL/dz_j = grad_out * (1[j == y] - exp(z_j - lse))."""
        row = tl.program_id(0)
        row_ptr = logits_ptr + row * stride_row
        grad_row_ptr = grad_logits_ptr + row * stride_row
        target = tl.load(targets_ptr + row)
        lse = tl.load(lse_ptr + row).to(tl.float32)
        g_out = tl.load(grad_out_ptr + row).to(tl.float32)

        for col_start in range(0, vocab_size, BLOCK_V):
            cols = col_start + tl.arange(0, BLOCK_V)
            mask = cols < vocab_size
            z = tl.load(row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
            prob = tl.exp(z - lse)
            is_target = (cols == target).to(tl.float32)
            gz = g_out * (is_target - prob)
            tl.store(grad_row_ptr + cols, gz.to(logits_ptr.dtype.element_ty), mask=mask)


class _FusedLogProbFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, logits_2d: torch.Tensor, targets_1d: torch.Tensor) -> torch.Tensor:
        n_rows, vocab = logits_2d.shape
        out_lp = torch.empty((n_rows,), device=logits_2d.device, dtype=torch.float32)
        out_lse = torch.empty((n_rows,), device=logits_2d.device, dtype=torch.float32)
        block_v = 4096
        _fused_logprob_fwd_kernel[(n_rows,)](
            logits_2d,
            targets_1d,
            out_lp,
            out_lse,
            logits_2d.stride(0),
            vocab,
            BLOCK_V=block_v,
        )
        ctx.save_for_backward(logits_2d, targets_1d, out_lse)
        return out_lp

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, None]:
        logits_2d, targets_1d, lse = ctx.saved_tensors
        n_rows, vocab = logits_2d.shape
        grad_logits = torch.empty_like(logits_2d)
        block_v = 4096
        _fused_logprob_bwd_kernel[(n_rows,)](
            logits_2d,
            targets_1d,
            lse,
            grad_out.contiguous(),
            grad_logits,
            logits_2d.stride(0),
            vocab,
            BLOCK_V=block_v,
        )
        return grad_logits, None


def fused_shifted_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Compute shifted per-token log P(x_t | x_<t) using Triton online-LSE on CUDA.

    Falls back to standard PyTorch `-F.cross_entropy` on CPU/MPS or when Triton is unavailable.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    bsz, seq_len, vocab = shift_logits.shape
    flat_logits = shift_logits.view(bsz * seq_len, vocab)
    flat_labels = shift_labels.view(bsz * seq_len)

    if HAS_TRITON and flat_logits.is_cuda:
        lp: torch.Tensor = _FusedLogProbFn.apply(flat_logits, flat_labels)
        return lp.view(bsz, seq_len)

    neg_nll = -F.cross_entropy(flat_logits.float(), flat_labels, reduction="none")
    return neg_nll.view(bsz, seq_len)


def benchmark_triton_vs_pytorch(
    *,
    batch_size: int = 2,
    seq_len: int = 2048,
    vocab_size: int = 152_064,
    warmup: int = 3,
    iters: int = 10,
) -> dict[str, Any]:
    """Benchmark fused Triton online-LSE logprobs vs eager PyTorch on Qwen's 152k vocab."""
    if not torch.cuda.is_available() or not HAS_TRITON:
        raise RuntimeError("CUDA + Triton required for GPU kernel benchmark")

    device = torch.device("cuda")
    torch.manual_seed(0)
    base_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=torch.bfloat16)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # 1. Exact numerical parity check (forward + backward)
    l_ref = base_logits.clone().detach().requires_grad_(True)
    lp_ref = -F.cross_entropy(
        l_ref[:, :-1, :].reshape(-1, vocab_size).float(),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).view(batch_size, seq_len - 1)
    lp_ref.mean().backward()  # type: ignore[no-untyped-call]
    g_ref = l_ref.grad
    assert g_ref is not None

    l_tri = base_logits.clone().detach().requires_grad_(True)
    lp_tri = fused_shifted_logprobs(l_tri, input_ids)
    lp_tri.mean().backward()  # type: ignore[no-untyped-call]
    g_tri = l_tri.grad
    assert g_tri is not None

    fwd_max_err = float((lp_ref - lp_tri).abs().max().item())
    bwd_max_err = float((g_ref.float() - g_tri.float()).abs().max().item())

    del l_ref, lp_ref, g_ref, l_tri, lp_tri, g_tri
    torch.cuda.empty_cache()

    def _measure(use_triton: bool) -> tuple[float, float]:
        for _ in range(warmup):
            x = base_logits.detach().clone().requires_grad_(True)
            out = (
                fused_shifted_logprobs(x, input_ids)
                if use_triton
                else -F.cross_entropy(
                    x[:, :-1, :].reshape(-1, vocab_size).float(),
                    input_ids[:, 1:].reshape(-1),
                    reduction="none",
                )
            )
            out.mean().backward()  # type: ignore[no-untyped-call]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            x = base_logits.detach().clone().requires_grad_(True)
            out = (
                fused_shifted_logprobs(x, input_ids)
                if use_triton
                else -F.cross_entropy(
                    x[:, :-1, :].reshape(-1, vocab_size).float(),
                    input_ids[:, 1:].reshape(-1),
                    reduction="none",
                )
            )
            out.mean().backward()  # type: ignore[no-untyped-call]
        torch.cuda.synchronize()
        ms = ((time.perf_counter() - t0) * 1000.0) / iters
        peak_gb = torch.cuda.max_memory_allocated() / (1024.0**3)
        return ms, peak_gb

    pt_ms, pt_gb = _measure(use_triton=False)
    tri_ms, tri_gb = _measure(use_triton=True)

    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "shape": [batch_size, seq_len, vocab_size],
        "dtype": "bfloat16",
        "fwd_max_abs_err": fwd_max_err,
        "bwd_max_abs_err": bwd_max_err,
        "pytorch_ms": round(pt_ms, 2),
        "triton_ms": round(tri_ms, 2),
        "speedup_x": round(pt_ms / max(tri_ms, 1e-6), 2),
        "pytorch_peak_gb": round(pt_gb, 3),
        "triton_peak_gb": round(tri_gb, 3),
        "vram_saved_gb": round(pt_gb - tri_gb, 3),
    }
