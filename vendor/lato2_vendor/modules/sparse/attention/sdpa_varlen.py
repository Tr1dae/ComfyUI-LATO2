# ComfyUI-LATO2: variable-length attention on top of torch SDPA, replacing the
# flash_attn varlen kernels used upstream (no flash-attn build needed).

from typing import List

import torch
import torch.nn.functional as F

# Cap on padded (num_seqs * max_len) tokens per SDPA call.
_MAX_PADDED_TOKENS = 1 << 17


def _pad(x: torch.Tensor, seqlens: List[int], max_len: int):
    """(T, H, C) packed -> (B, H, L, C) padded, plus (B, L) validity mask."""
    B = len(seqlens)
    H, C = x.shape[1], x.shape[2]
    lens = torch.tensor(seqlens, device=x.device)
    batch = torch.repeat_interleave(torch.arange(B, device=x.device), lens)
    starts = torch.cumsum(lens, 0) - lens
    pos = torch.arange(x.shape[0], device=x.device) - starts[batch]
    out = x.new_zeros((B, max_len, H, C))
    out[batch, pos] = x
    mask = torch.arange(max_len, device=x.device)[None] < lens[:, None]
    return out.transpose(1, 2), mask, batch, pos


def varlen_attention(q, k, v, q_seqlen: List[int], kv_seqlen: List[int]) -> torch.Tensor:
    """q: (Tq, H, C), k/v: (Tk, H, C); sequences packed back to back. Returns (Tq, H, Cv)."""
    n = len(q_seqlen)
    if n == 1:
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None]
        )
        return o[0].transpose(0, 1)

    if all(l == q_seqlen[0] for l in q_seqlen) and all(l == kv_seqlen[0] for l in kv_seqlen):
        lq, lk = q_seqlen[0], kv_seqlen[0]
        qq = q.view(n, lq, *q.shape[1:]).transpose(1, 2)
        kk = k.view(n, lk, *k.shape[1:]).transpose(1, 2)
        vv = v.view(n, lk, *v.shape[1:]).transpose(1, 2)
        o = F.scaled_dot_product_attention(qq, kk, vv)
        return o.transpose(1, 2).reshape(-1, *o.shape[1:2], o.shape[-1])

    out = torch.empty((q.shape[0], q.shape[1], v.shape[2]), dtype=q.dtype, device=q.device)
    q_starts = [0]
    for l in q_seqlen:
        q_starts.append(q_starts[-1] + l)
    k_starts = [0]
    for l in kv_seqlen:
        k_starts.append(k_starts[-1] + l)

    # group sequences so the padded batch stays bounded
    i = 0
    while i < n:
        j = i
        mq = mk = 0
        while j < n:
            nq, nk = max(mq, q_seqlen[j]), max(mk, kv_seqlen[j])
            if j > i and (j - i + 1) * max(nq, nk) > _MAX_PADDED_TOKENS:
                break
            mq, mk = nq, nk
            j += 1
        qs = q[q_starts[i]:q_starts[j]]
        ks = k[k_starts[i]:k_starts[j]]
        vs = v[k_starts[i]:k_starts[j]]
        qp, _, qb, qpos = _pad(qs, q_seqlen[i:j], mq)
        kp, kmask, _, _ = _pad(ks, kv_seqlen[i:j], mk)
        vp, _, _, _ = _pad(vs, kv_seqlen[i:j], mk)
        attn_mask = kmask[:, None, None, :]  # (B, 1, 1, Lk)
        o = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=attn_mask)  # (B, H, Lq, C)
        o = o.transpose(1, 2)  # (B, Lq, H, C)
        out[q_starts[i]:q_starts[j]] = o[qb, qpos]
        i = j
    return out
