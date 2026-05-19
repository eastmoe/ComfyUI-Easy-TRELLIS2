"""
Attention dispatch for TRELLIS2 dense and sparse/varlen tensors.

This prefers optional accelerated backends when available, but keeps ComfyUI
attention and PyTorch SDPA as first-class fallbacks so flash-attn/sageattention
are not required to run the nodes.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn.functional as F

log = logging.getLogger("trellis2")

_ATTN_BACKEND = "auto"
_dense_printed = False
_varlen_printed = False

_BACKEND_ALIASES = {
    "auto": "auto",
    "comfy": "comfy",
    "comfyui": "comfy",
    "pytorch": "pytorch",
    "torch": "pytorch",
    "sdpa": "pytorch",
    "naive": "pytorch",
    "xformers": "xformers",
    "flash": "flash_attn",
    "flash_attn": "flash_attn",
    "sage": "sageattn",
    "sageattn": "sageattn",
}

__all__ = [
    "scaled_dot_product_attention",
    "sparse_scaled_dot_product_attention",
    "dispatch_varlen_attention",
    "set_attn_backend",
    "get_attn_backend",
]


def set_attn_backend(backend: str) -> None:
    """Set the preferred TRELLIS2 attention backend."""
    global _ATTN_BACKEND, _dense_printed, _varlen_printed
    normalized = _BACKEND_ALIASES.get(backend, "auto")
    if normalized != backend:
        log.info("Attention backend '%s' normalized to '%s'", backend, normalized)
    _ATTN_BACKEND = normalized
    _dense_printed = False
    _varlen_printed = False


def get_attn_backend() -> str:
    return _ATTN_BACKEND


def _parse_dense_args(*args, **kwargs):
    transformer_options = kwargs.pop("transformer_options", {})
    arg_names = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names, (
        f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    )
    for key in arg_names[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if args else kwargs["qkv"]
        assert len(qkv.shape) == 5 and qkv.shape[2] == 3, (
            f"Invalid shape for qkv, got {qkv.shape}, expected [N, L, 3, H, C]"
        )
        q, k, v = qkv.unbind(dim=2)
    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs["q"]
        kv = args[1] if len(args) > 1 else kwargs["kv"]
        assert q.shape[0] == kv.shape[0], (
            f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        )
        assert len(q.shape) == 4, (
            f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
        )
        assert len(kv.shape) == 5 and kv.shape[2] == 2, (
            f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
        )
        k, v = kv.unbind(dim=2)
    else:
        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]
        assert q.shape[0] == k.shape[0] == v.shape[0], (
            f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        )
        assert len(q.shape) == 4, (
            f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
        )
        assert len(k.shape) == 4, (
            f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
        )
        assert len(v.shape) == 4, (
            f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
        )

    return q, k, v, transformer_options


def _pytorch_attention_nlhc(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    return out.permute(0, 2, 1, 3)


def _comfy_attention_function(device: torch.device) -> Callable:
    from comfy.ldm.modules.attention import optimized_attention_for_device

    return optimized_attention_for_device(device)


def _named_comfy_attention_function(name: str) -> Callable | None:
    from comfy.ldm.modules.attention import get_attention_function

    comfy_names = {
        "xformers": "xformers",
        "flash_attn": "flash",
        "sageattn": "sage",
    }
    return get_attention_function(comfy_names[name], default=None)


def _comfy_attention_nlhc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    transformer_options=None,
    preferred: str = "auto",
) -> torch.Tensor:
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    heads = q.shape[1]

    attn_fn = None
    if preferred in ("xformers", "flash_attn", "sageattn"):
        attn_fn = _named_comfy_attention_function(preferred)
        if attn_fn is None:
            log.warning(
                "%s attention requested but unavailable; falling back to ComfyUI attention",
                preferred,
            )
    if attn_fn is None:
        attn_fn = _comfy_attention_function(q.device)

    out = attn_fn(
        q,
        k,
        v,
        heads=heads,
        skip_reshape=True,
        skip_output_reshape=True,
        transformer_options=transformer_options or {},
    )
    return out.permute(0, 2, 1, 3)


def scaled_dot_product_attention(*args, **kwargs):
    """
    TRELLIS2 dense attention.

    Input layout is [N, L, H, C] or packed qkv/kv variants. Output layout is
    [N, L, H, C].
    """
    global _dense_printed

    q, k, v, transformer_options = _parse_dense_args(*args, **kwargs)
    backend = get_attn_backend()
    if not _dense_printed:
        log.info("Dense attention backend preference: %s", backend)
        _dense_printed = True

    if backend == "pytorch":
        return _pytorch_attention_nlhc(q, k, v)

    try:
        return _comfy_attention_nlhc(q, k, v, transformer_options, preferred=backend)
    except Exception as exc:
        if backend in ("flash_attn", "sageattn", "xformers", "auto", "comfy"):
            log.warning(
                "ComfyUI/%s dense attention failed (%s); using PyTorch SDPA fallback",
                backend,
                exc,
            )
            return _pytorch_attention_nlhc(q, k, v)
        raise


def _sdpa_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
) -> torch.Tensor:
    del max_seqlen_q, max_seqlen_kv
    batch = cu_seqlens_q.shape[0] - 1
    total_q = q.shape[0]
    total_kv = k.shape[0]

    mask = torch.full((total_q, total_kv), float("-inf"), device=q.device, dtype=q.dtype)
    for i in range(batch):
        qs, qe = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        ks, ke = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()
        mask[qs:qe, ks:ke] = 0.0

    out = F.scaled_dot_product_attention(
        q.unsqueeze(0).permute(0, 2, 1, 3),
        k.unsqueeze(0).permute(0, 2, 1, 3),
        v.unsqueeze(0).permute(0, 2, 1, 3),
        attn_mask=mask.unsqueeze(0).unsqueeze(0),
        dropout_p=0.0,
        is_causal=False,
    )
    return out.permute(0, 2, 1, 3).squeeze(0)


def _comfy_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
) -> torch.Tensor:
    del max_seqlen_q, max_seqlen_kv
    attn_fn = _comfy_attention_function(q.device)
    heads = q.shape[1]
    out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], device=q.device, dtype=v.dtype)

    for i in range(cu_seqlens_q.shape[0] - 1):
        qs, qe = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        ks, ke = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()
        qi = q[qs:qe].unsqueeze(0).permute(0, 2, 1, 3)
        ki = k[ks:ke].unsqueeze(0).permute(0, 2, 1, 3)
        vi = v[ks:ke].unsqueeze(0).permute(0, 2, 1, 3)
        oi = attn_fn(
            qi,
            ki,
            vi,
            heads=heads,
            skip_reshape=True,
            skip_output_reshape=True,
        )
        out[qs:qe] = oi.permute(0, 2, 1, 3).squeeze(0)
    return out


def _accelerated_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
) -> torch.Tensor:
    from comfy_sparse_attn import dispatch_varlen_attention as package_dispatch

    return package_dispatch(
        q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
    )


def dispatch_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
) -> torch.Tensor:
    """Variable-length attention with ComfyUI/PyTorch fallback paths."""
    global _varlen_printed

    backend = get_attn_backend()
    if not _varlen_printed:
        log.info("Varlen attention backend preference: %s", backend)
        _varlen_printed = True

    if backend == "pytorch":
        return _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
    if backend == "comfy":
        return _comfy_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)

    try:
        return _accelerated_varlen(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
        )
    except Exception as exc:
        log.warning(
            "%s varlen attention failed or is unavailable (%s); using ComfyUI fallback",
            backend,
            exc,
        )
        try:
            return _comfy_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
            )
        except Exception as comfy_exc:
            log.warning("ComfyUI varlen fallback failed (%s); using PyTorch SDPA", comfy_exc)
            return _sdpa_varlen(
                q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
            )


def sparse_scaled_dot_product_attention(*args, **kwargs):
    """Sparse/variable-length attention for VarLenTensor and dense mixes."""
    from .sparse import VarLenTensor

    arg_names = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names, (
        f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    )
    for key in arg_names[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if args else kwargs["qkv"]
        assert isinstance(qkv, VarLenTensor), f"qkv must be a VarLenTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, (
            f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        )
        device = qkv.device
        structure = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        q, k, v = qkv.feats.unbind(dim=1)
        dense_shape = None

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs["q"]
        kv = args[1] if len(args) > 1 else kwargs["kv"]
        assert (
            isinstance(q, VarLenTensor) and isinstance(kv, (VarLenTensor, torch.Tensor))
        ) or (
            isinstance(q, torch.Tensor) and isinstance(kv, VarLenTensor)
        ), f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], (
            f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        )
        device = q.device

        if isinstance(q, VarLenTensor):
            assert len(q.shape) == 3, (
                f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            )
            structure = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats
            dense_shape = None
        else:
            assert len(q.shape) == 4, (
                f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            )
            structure = None
            dense_shape = q.shape
            batch, length, heads, channels = q.shape
            q_seqlen = [length] * batch
            q = q.reshape(batch * length, heads, channels)

        if isinstance(kv, VarLenTensor):
            assert len(kv.shape) == 4 and kv.shape[1] == 2, (
                f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            )
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv_feats = kv.feats
        else:
            assert len(kv.shape) == 5 and kv.shape[2] == 2, (
                f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            )
            batch, length, _, heads, channels = kv.shape
            kv_seqlen = [length] * batch
            kv_feats = kv.reshape(batch * length, 2, heads, channels)
        k, v = kv_feats.unbind(dim=1)

    else:
        q = args[0] if len(args) > 0 else kwargs["q"]
        k = args[1] if len(args) > 1 else kwargs["k"]
        v = args[2] if len(args) > 2 else kwargs["v"]
        assert (
            isinstance(q, VarLenTensor)
            and isinstance(k, (VarLenTensor, torch.Tensor))
            and type(k) is type(v)
        ) or (
            isinstance(q, torch.Tensor)
            and isinstance(k, VarLenTensor)
            and isinstance(v, VarLenTensor)
        ), f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert q.shape[0] == k.shape[0] == v.shape[0], (
            f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        )
        device = q.device

        if isinstance(q, VarLenTensor):
            assert len(q.shape) == 3, (
                f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            )
            structure = q
            dense_shape = None
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats
        else:
            assert len(q.shape) == 4, (
                f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            )
            structure = None
            dense_shape = q.shape
            batch, length, heads, channels = q.shape
            q_seqlen = [length] * batch
            q = q.reshape(batch * length, heads, channels)

        if isinstance(k, VarLenTensor):
            assert len(k.shape) == 3, (
                f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            )
            assert len(v.shape) == 3, (
                f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            )
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k = k.feats
            v = v.feats
        else:
            assert len(k.shape) == 4, (
                f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            )
            assert len(v.shape) == 4, (
                f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            )
            batch, length, heads, channels = k.shape
            kv_seqlen = [length] * batch
            k = k.reshape(batch * length, heads, channels)
            v = v.reshape(batch * length, heads, v.shape[-1])

    cu_seqlens_q = torch.cat(
        [torch.tensor([0], device=device), torch.cumsum(torch.tensor(q_seqlen, device=device), dim=0)]
    ).int()
    cu_seqlens_kv = torch.cat(
        [torch.tensor([0], device=device), torch.cumsum(torch.tensor(kv_seqlen, device=device), dim=0)]
    ).int()

    out = dispatch_varlen_attention(
        q, k, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen)
    )

    if structure is not None:
        return structure.replace(out)

    batch, length, heads, _channels = dense_shape
    return out.reshape(batch, length, heads, -1)
