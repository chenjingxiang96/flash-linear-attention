# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import torch

from fla.utils.ascendc import verify_ascendc_tensors


def _head_first(tensor):
    return None if tensor is None else tensor.transpose(1, 2).contiguous()


def chunk_scaled_dot_kkt_fwd_verifier(
    k, g=None, beta=None, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32, chunk_indices=None,
):
    supported, reason = verify_ascendc_tensors(k, g, beta)
    if not supported:
        return supported, reason
    if cu_seqlens is not None or chunk_indices is not None:
        return False, 'AscendC KKT currently supports dense inputs'
    if k.ndim != 4 or k.shape[-1] != 128 or chunk_size != 64:
        return False, 'AscendC KKT requires rank 4, K=128 and chunk_size=64'
    if g is None or beta is None or g.ndim != 3 or beta.shape != g.shape:
        return False, 'AscendC KKT requires scalar gate and beta [B, T, HV]'
    if g.shape[:2] != k.shape[:2] or g.shape[2] % k.shape[2]:
        return False, 'AscendC KKT requires matching B/T and HV divisible by HK'
    if k.dtype not in (torch.float16, torch.bfloat16) or g.dtype != torch.float32 or beta.dtype != torch.float32:
        return False, 'AscendC KKT requires half-precision keys and float32 gate/beta'
    if output_dtype != torch.float32:
        return False, 'AscendC KKT produces float32 output'
    return True, None


def chunk_scaled_dot_kkt_fwd(
    k, g=None, beta=None, cu_seqlens=None, chunk_size=64, output_dtype=torch.float32, chunk_indices=None,
):
    from fla_npu.ops.ascendc import chunk_scaled_dot_kkt

    result = chunk_scaled_dot_kkt(
        k=_head_first(k),
        g=_head_first(g * math.log(2)),
        beta=_head_first(beta),
        chunk_size=chunk_size,
    )
    return _head_first(result)


def chunk_gated_delta_rule_fwd_h_verifier(
    k, w, u, g=None, gk=None, initial_state=None, output_final_state=False, chunk_size=64,
    save_new_value=True, state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None, chunk_offsets=None,
):
    supported, reason = verify_ascendc_tensors(k, w, u, g, gk, initial_state)
    if not supported:
        return supported, reason
    if any(x is not None for x in (cu_seqlens, cu_seqlens_cpu, chunk_indices, chunk_offsets)):
        return False, 'AscendC state update currently supports dense inputs'
    if chunk_size != 64 or not save_new_value or state_v_first:
        return False, 'AscendC state update requires chunk_size=64, save_new_value and key-first states'
    if any(x.ndim != 4 or x.dtype != torch.bfloat16 or x.shape[-1] != 128 for x in (k, w, u)):
        return False, 'AscendC state update requires bfloat16 rank-4 tensors with K=V=128'
    if w.shape != u.shape or k.shape[:2] != u.shape[:2] or u.shape[2] % k.shape[2]:
        return False, 'AscendC state update requires matching B/T and grouped value heads'
    if (g is None) == (gk is None):
        return False, 'AscendC state update requires exactly one of g/gk'
    gate = g if g is not None else gk
    if gate.shape != (u.shape[:3] if g is not None else u.shape) or gate.dtype not in (torch.float32, torch.bfloat16):
        return False, 'AscendC state update requires a gate matching the value heads'
    if gk is not None and k.shape[2] != u.shape[2]:
        return False, 'AscendC vector gate requires prepared keys with HV heads'
    if initial_state is not None:
        if initial_state.shape != (k.shape[0], u.shape[2], 128, 128) or initial_state.dtype != torch.float32:
            return False, 'AscendC initial state must be float32 [B, HV, K, V]'
    return True, None


def chunk_gated_delta_rule_fwd_h(
    k, w, u, g=None, gk=None, initial_state=None, output_final_state=False, chunk_size=64,
    save_new_value=True, state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None, chunk_offsets=None,
):
    from fla_npu.ops.ascendc import chunk_fwd_h

    h, v_new, final_state = chunk_fwd_h(
        k=_head_first(k),
        w=_head_first(w),
        u=_head_first(u),
        g=_head_first(g),
        gk=_head_first(gk),
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=save_new_value,
        use_exp2=True,
        state_v_first=state_v_first,
    )
    return h, _head_first(v_new), final_state


def chunk_fwd_o_verifier(
    q, k, v, h, g=None, g_gamma=None, scale=None, state_v_first=False,
    cu_seqlens=None, chunk_size=64, chunk_indices=None,
):
    supported, reason = verify_ascendc_tensors(q, k, v, h, g)
    if not supported:
        return supported, reason
    if cu_seqlens is not None or chunk_indices is not None or g_gamma is not None or state_v_first:
        return False, 'AscendC output currently requires dense input, scalar gates and key-first states'
    if chunk_size != 64 or q.shape != k.shape or q.ndim != 4 or v.ndim != 4:
        return False, 'AscendC output requires matching Q/K and chunk_size=64'
    if q.shape[-1] != 128 or v.shape[-1] != 128 or q.shape[:2] != v.shape[:2] or v.shape[2] % q.shape[2]:
        return False, 'AscendC output requires K=V=128 and grouped value heads'
    if any(x.dtype != torch.bfloat16 for x in (q, k, v, h)):
        return False, 'AscendC output requires bfloat16 Q/K/V/states'
    if g is None or g.shape != v.shape[:3] or g.dtype != torch.float32:
        return False, 'AscendC output requires a float32 scalar gate [B, T, HV]'
    if h.shape != (q.shape[0], math.ceil(q.shape[1] / chunk_size), v.shape[2], 128, 128):
        return False, 'AscendC output requires states [B, NT, HV, K, V]'
    return True, None


def chunk_fwd_o(
    q, k, v, h, g=None, g_gamma=None, scale=None, state_v_first=False,
    cu_seqlens=None, chunk_size=64, chunk_indices=None,
):
    from fla_npu.ops.ascendc import chunk_fwd_o as native_output

    return native_output(
        q=_head_first(q),
        k=_head_first(k),
        v=_head_first(v),
        h=h.contiguous(),
        scale=q.shape[-1] ** -0.5 if scale is None else scale,
        g=_head_first(g),
        chunk_size=chunk_size,
        use_exp2=True,
        output_layout='BSND',
    )


def chunk_bwd_dv_local_verifier(
    q, k, do, g=None, g_gamma=None, A=None, scale=None, cu_seqlens=None, chunk_size=64, chunk_indices=None,
):
    supported, reason = verify_ascendc_tensors(q, k, do, g)
    if not supported:
        return supported, reason
    if any(x is not None for x in (g_gamma, A, cu_seqlens, chunk_indices)) or chunk_size != 64:
        return False, 'AscendC local value gradient requires dense scalar-gated input and chunk_size=64'
    if q.ndim != 4 or k.shape != q.shape or do.shape != q.shape or q.shape[-1] != 128:
        return False, 'AscendC local value gradient requires equal heads and K=V=128'
    if any(x.dtype != torch.bfloat16 for x in (q, k, do)):
        return False, 'AscendC local value gradient requires bfloat16 Q/K/dO'
    if g is None or g.shape != q.shape[:3] or g.dtype != torch.float32:
        return False, 'AscendC local value gradient requires float32 scalar gates'
    return True, None


def chunk_bwd_dv_local(
    q, k, do, g=None, g_gamma=None, A=None, scale=None, cu_seqlens=None, chunk_size=64, chunk_indices=None,
):
    from fla_npu.ops.ascendc import chunk_bwd_dv_local as native_value_gradient

    result = native_value_gradient(
        q=_head_first(q),
        k=_head_first(k),
        d_o=_head_first(do),
        g=_head_first(g * math.log(2)),
        scale=q.shape[-1] ** -0.5 if scale is None else scale,
        chunk_size=chunk_size,
    )
    return _head_first(result)


def chunk_bwd_dqkwg_verifier(
    q, k, v, do, h, dh, w=None, g=None, g_gamma=None, dv=None, scale=None,
    state_v_first=False, cu_seqlens=None, chunk_size=64, chunk_indices=None,
):
    supported, reason = chunk_fwd_o_verifier(
        q, k, v, h, g, g_gamma, scale, state_v_first, cu_seqlens, chunk_size, chunk_indices,
    )
    if not supported:
        return supported, reason
    if q.shape != v.shape or w is None or dv is None:
        return False, 'AscendC Q/K/W gradient requires equal heads, W and dV'
    for x, reference in ((do, v), (dh, h), (w, k), (dv, v)):
        if x.shape != reference.shape or x.dtype != reference.dtype or x.device != reference.device:
            return False, 'AscendC Q/K/W gradient requires gradients matching the input tensors'
    return True, None


def chunk_bwd_dqkwg(
    q, k, v, do, h, dh, w=None, g=None, g_gamma=None, dv=None, scale=None,
    state_v_first=False, cu_seqlens=None, chunk_size=64, chunk_indices=None,
):
    from fla_npu.ops.ascendc import chunk_bwd_dqkwg as native_qkw_gradient

    results = native_qkw_gradient(
        q=_head_first(q),
        k=_head_first(k),
        v=_head_first(v),
        g=_head_first(g * math.log(2)),
        h=h.contiguous(),
        dox=_head_first(do),
        dh=dh.contiguous(),
        dv=_head_first(dv),
        chunk_size=chunk_size,
        scale=q.shape[-1] ** -0.5 if scale is None else scale,
        use_exp2=False,
        transpose_state_layout=False,
    )
    return tuple(_head_first(x) for x in results)


def chunk_gated_delta_rule_bwd_dhu_verifier(
    q, k, w, do, dv, g=None, gk=None, h0=None, dht=None, scale=None,
    state_v_first=False, cu_seqlens=None, chunk_size=64, chunk_indices=None, chunk_offsets=None, use_graph=False,
):
    supported, reason = chunk_bwd_dv_local_verifier(
        q,
        k,
        do,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    if not supported:
        return supported, reason
    if any(x is not None for x in (gk, h0, dht, chunk_offsets)) or use_graph or state_v_first:
        return False, 'AscendC state gradient currently requires dense scalar gates without boundary states'
    if any(x.shape != q.shape or x.dtype != q.dtype or x.device != q.device for x in (w, dv)):
        return False, 'AscendC state gradient requires W/dV matching Q/K'
    return True, None


def chunk_gated_delta_rule_bwd_dhu(
    q, k, w, do, dv, g=None, gk=None, h0=None, dht=None, scale=None,
    state_v_first=False, cu_seqlens=None, chunk_size=64, chunk_indices=None, chunk_offsets=None, use_graph=False,
):
    from fla_npu.ops.ascendc import chunk_gated_delta_rule_bwd_dhu as native_state_gradient

    dh, dh0, dv2 = native_state_gradient(
        q=_head_first(q),
        k=_head_first(k),
        w=_head_first(w),
        d_o=_head_first(do),
        dv=_head_first(dv),
        scale=q.shape[-1] ** -0.5 if scale is None else scale,
        chunk_size=chunk_size,
        g=_head_first(g),
        use_exp2=True,
    )
    return dh, dh0, _head_first(dv2)
