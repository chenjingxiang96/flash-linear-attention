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


def recompute_w_u_fwd_verifier(k, v, beta, A, g=None, cu_seqlens=None, chunk_indices=None):
    supported, reason = verify_ascendc_tensors(k, v, beta, A, g)
    if not supported:
        return supported, reason
    if cu_seqlens is not None or chunk_indices is not None:
        return False, 'AscendC WY currently supports dense inputs'
    if k.ndim != 4 or v.ndim != 4 or k.shape != v.shape or k.shape[-1] != 128:
        return False, 'AscendC WY currently requires equal heads and K=V=128'
    if k.dtype not in (torch.float16, torch.bfloat16) or v.dtype != k.dtype or A.dtype != k.dtype:
        return False, 'AscendC WY requires matching half-precision K/V/A'
    if A.shape != (*v.shape[:3], 64):
        return False, 'AscendC WY requires A [B, T, H, 64]'
    if g is None or beta.shape != v.shape[:3] or g.shape != beta.shape:
        return False, 'AscendC WY requires scalar gate and beta [B, T, H]'
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        return False, 'AscendC WY requires float32 gate/beta'
    return True, None


def recompute_w_u_fwd(k, v, beta, A, g=None, cu_seqlens=None, chunk_indices=None):
    from fla_npu.ops.ascendc import recompute_w_u_fwd as native_wy

    w, u = native_wy(
        k=_head_first(k),
        v=_head_first(v),
        beta=_head_first(beta),
        A=_head_first(A),
        chunk_size=A.shape[-1],
        g=_head_first(g * math.log(2)),
    )
    return _head_first(w), _head_first(u)


def prepare_wy_repr_bwd_verifier(k, v, beta, A, dw, du, g=None, cu_seqlens=None, chunk_indices=None):
    supported, reason = recompute_w_u_fwd_verifier(k, v, beta, A, g, cu_seqlens, chunk_indices)
    if not supported:
        return supported, reason
    if any(x.device != k.device or x.shape != k.shape or x.dtype != k.dtype for x in (dw, du)):
        return False, 'AscendC WY gradients must match K/V shape, dtype and device'
    return True, None


def prepare_wy_repr_bwd(k, v, beta, A, dw, du, g=None, cu_seqlens=None, chunk_indices=None):
    from fla_npu.ops.ascendc import prepare_wy_repr_bwd as native_wy_backward

    results = native_wy_backward(
        k=_head_first(k),
        v=_head_first(v),
        beta=_head_first(beta),
        A=_head_first(A),
        dw=_head_first(dw),
        du=_head_first(du),
        g=_head_first(g * math.log(2)),
        chunk_size=A.shape[-1],
    )
    return tuple(_head_first(x) for x in results)


def chunk_gated_delta_rule_verifier(
    q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
    state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None, cp_context=None, **kwargs,
):
    supported, reason = verify_ascendc_tensors(q, k, v, g, beta, initial_state)
    if not supported:
        return supported, reason
    if any(x is not None for x in (cu_seqlens, cu_seqlens_cpu, chunk_indices, cp_context)) or kwargs:
        return False, 'AscendC fused GDN currently supports dense inference without context parallelism'
    if use_qk_l2norm_in_kernel or use_beta_sigmoid_in_kernel or allow_neg_eigval:
        return False, 'AscendC fused GDN currently requires preprocessed Q/K/beta'
    if q.ndim != 4 or q.shape != k.shape or k.shape != v.shape or q.shape[-1] != 128:
        return False, 'AscendC fused GDN currently requires equal heads and K=V=128'
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        return False, 'AscendC fused GDN requires bfloat16 Q/K/V'
    if g.shape != q.shape[:3] or beta.shape != g.shape or g.dtype != torch.float32 or beta.dtype != torch.float32:
        return False, 'AscendC fused GDN requires float32 scalar gate/beta'
    if initial_state is not None:
        if initial_state.shape != (q.shape[0], q.shape[2], 128, 128) or initial_state.dtype != torch.float32:
            return False, 'AscendC fused GDN requires float32 initial state [B, H, K, V]'
    return True, None


def chunk_gated_delta_rule(
    q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
    state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None, cp_context=None, **kwargs,
):
    from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd

    result = chunk_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        initial_state=None if initial_state is None else initial_state.clone(),
        output_final_state=output_final_state,
        scale=scale,
        use_exp2=False,
        disable_recompute=False,
        state_v_first=state_v_first,
        layout='BSND',
    )
    return result[0], result[1]


def fused_recurrent_gated_delta_rule_fwd_verifier(
    q, k, v, g=None, gk=None, gv=None, beta=None, A_log=None, dt_bias=None, scale=None,
    initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, state_v_first=False, cu_seqlens=None,
):
    supported, reason = verify_ascendc_tensors(q, k, v, g, beta, initial_state)
    if not supported:
        return supported, reason
    if any(x is not None for x in (gk, gv, A_log, dt_bias, cu_seqlens)):
        return False, 'AscendC recurrent GDN currently supports dense scalar gates'
    if use_qk_l2norm_in_kernel or use_beta_sigmoid_in_kernel or allow_neg_eigval:
        return False, 'AscendC recurrent GDN requires preprocessed Q/K/beta'
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[-1] != 128 or q.shape[2] > 256:
        return False, 'AscendC recurrent GDN requires equal heads, H<=256 and K=V=128'
    if q.shape[1] > 8:
        return False, 'AscendC recurrent GDN supports at most 8 tokens per sequence'
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        return False, 'AscendC recurrent GDN requires bfloat16 Q/K/V'
    if beta is None or beta.shape != q.shape[:3] or beta.dtype != torch.bfloat16:
        return False, 'AscendC recurrent GDN requires bfloat16 beta [B, T, H]'
    if g is None or g.shape != q.shape[:3] or g.dtype != torch.float32:
        return False, 'AscendC recurrent GDN requires float32 gates [B, T, H]'
    if initial_state is not None:
        if initial_state.shape != (q.shape[0], q.shape[2], 128, 128) or initial_state.dtype != torch.float32:
            return False, 'AscendC recurrent GDN requires float32 states'
    return True, None


def fused_recurrent_gated_delta_rule_fwd(
    q, k, v, g=None, gk=None, gv=None, beta=None, A_log=None, dt_bias=None, scale=None,
    initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, state_v_first=False, cu_seqlens=None,
):
    from fla_npu.ops.ascendc import recurrent_gated_delta_rule

    batch, tokens, heads, dim = q.shape
    # the native cache stores one state per token and requires unique slot indices.
    state = q.new_zeros(batch * tokens, heads, dim, dim, dtype=torch.float32)
    if initial_state is not None:
        state[::tokens].copy_(initial_state if state_v_first else initial_state.transpose(-1, -2))
    output = recurrent_gated_delta_rule(
        query=q.reshape(-1, heads, dim).contiguous(),
        key=k.reshape(-1, heads, dim).contiguous(),
        value=v.reshape(-1, heads, dim).contiguous(),
        state=state,
        beta=beta.reshape(-1, heads).contiguous(),
        scale=float(dim ** -0.5 if scale is None else scale),
        actual_seq_lengths=torch.tensor([0] + [tokens] * batch, device=q.device, dtype=torch.int32),
        ssm_state_indices=torch.arange(batch * tokens, device=q.device, dtype=torch.int32),
        g=g.reshape(-1, heads).contiguous(),
    )
    final_state = None
    if output_final_state:
        final_state = state[tokens - 1::tokens]
        final_state = final_state.contiguous() if state_v_first else final_state.transpose(-1, -2).contiguous()
    return output.reshape(v.shape), final_state


def chunk_gated_delta_rule_bwd_verifier(
    q, k, v, g, beta, A, scale, initial_state, do, dht, state_v_first=False,
    cu_seqlens=None, cp_context=None, chunk_indices=None, use_gate_in_kernel=False,
    g_input=None, A_log=None, dt_bias=None, chunk_size=64,
):
    supported, reason = recompute_w_u_fwd_verifier(k, v, beta, A, g, cu_seqlens, chunk_indices)
    if not supported:
        return supported, reason
    if any(x is not None for x in (initial_state, dht, cp_context, g_input, A_log, dt_bias)):
        return False, 'AscendC fused GDN backward currently excludes boundary states and gate activation'
    if use_gate_in_kernel or state_v_first or chunk_size != 64:
        return False, 'AscendC fused GDN backward requires scalar cumulative gates and chunk_size=64'
    if k.dtype != torch.bfloat16:
        return False, 'AscendC fused GDN backward requires bfloat16'
    if any(x.shape != k.shape or x.dtype != k.dtype or x.device != k.device for x in (q, do)):
        return False, 'AscendC fused GDN backward requires matching Q/K/V/dO'
    return True, None


def chunk_gated_delta_rule_bwd(
    q, k, v, g, beta, A, scale, initial_state, do, dht, state_v_first=False,
    cu_seqlens=None, cp_context=None, chunk_indices=None, use_gate_in_kernel=False,
    g_input=None, A_log=None, dt_bias=None, chunk_size=64,
):
    from fla_npu.ops.ascendc import chunk_gated_delta_rule_bwd as native_backward

    return native_backward(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        A=_head_first(A),
        d_o=do.contiguous(),
        scale=scale,
        chunk_size=chunk_size,
        use_exp2=True,
        layout='BSND',
    )
