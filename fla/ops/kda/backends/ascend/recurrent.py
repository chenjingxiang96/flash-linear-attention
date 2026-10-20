# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.utils.ascendc import verify_ascendc_tensors


def fused_recurrent_kda_fwd_verifier(
    q, k, v, g, beta, A_log=None, dt_bias=None, initial_state=None, scale=None,
    output_final_state=False, inplace_final_state=True, state_v_first=False, cu_seqlens=None,
    ssm_state_indices=None, num_accepted_tokens=None, use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
    lower_bound=None, out=None, **kwargs,
):
    supported, reason = verify_ascendc_tensors(q, k, v, g, beta, A_log, dt_bias, initial_state, out)
    if not supported:
        return supported, reason
    if any(x is not None for x in (cu_seqlens, ssm_state_indices, num_accepted_tokens)) or kwargs:
        return False, 'AscendC recurrent KDA currently supports dense calls without state indexing'
    if q.ndim != 4 or q.shape != k.shape or v.ndim != 4 or q.shape[:2] != v.shape[:2]:
        return False, 'AscendC recurrent KDA requires matching dense Q/K/V'
    if q.shape[1] > 8:
        return False, 'AscendC recurrent KDA supports at most 8 tokens per sequence'
    if q.shape[-1] != 128 or v.shape[-1] not in (128, 256) or v.shape[2] % q.shape[2]:
        return False, 'AscendC recurrent KDA requires K=128, V=128/256 and grouped value heads'
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        return False, 'AscendC recurrent KDA requires bfloat16 Q/K/V'
    if g.shape != (*v.shape[:3], 128) or beta.shape != v.shape[:3]:
        return False, 'AscendC recurrent KDA requires gates and beta matching value heads'
    if any(x.dtype not in (torch.bfloat16, torch.float32) for x in (g, beta)):
        return False, 'AscendC recurrent KDA requires bfloat16/float32 gate and beta'
    if inplace_final_state and initial_state is None:
        return False, 'AscendC in-place recurrence requires an initial state'
    if initial_state is not None:
        tail = (v.shape[-1], 128) if state_v_first else (128, v.shape[-1])
        if initial_state.shape != (q.shape[0], v.shape[2], *tail) or initial_state.dtype != torch.float32:
            return False, 'AscendC recurrent KDA requires float32 states matching the selected layout'
        if not initial_state.is_contiguous():
            return False, 'AscendC recurrent KDA requires contiguous states'
        if inplace_final_state and initial_state.requires_grad:
            return False, 'AscendC in-place states must not require gradients'
    if use_gate_in_kernel:
        if A_log is None or A_log.shape != (v.shape[2],) or A_log.dtype != torch.float32:
            return False, 'AscendC fused recurrent gate requires float32 A_log [HV]'
        if dt_bias is not None and (dt_bias.shape != (v.shape[2] * 128,) or dt_bias.dtype != torch.float32):
            return False, 'AscendC fused recurrent gate requires float32 dt_bias [HV*K]'
        if lower_bound is not None and not -5 <= lower_bound < 0:
            return False, 'AscendC bounded recurrent gate requires lower_bound in [-5, 0)'
    elif A_log is not None or dt_bias is not None or lower_bound is not None:
        return False, 'AscendC recurrent gate parameters require use_gate_in_kernel'
    if out is not None and (out.shape != v.shape or out.dtype != v.dtype):
        return False, 'AscendC recurrent output must match V'
    return True, None


def fused_recurrent_kda_fwd(
    q, k, v, g, beta, A_log=None, dt_bias=None, initial_state=None, scale=None,
    output_final_state=False, inplace_final_state=True, state_v_first=False, cu_seqlens=None,
    ssm_state_indices=None, num_accepted_tokens=None, use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
    lower_bound=None, out=None, **kwargs,
):
    from fla_npu.ops.ascendc import recurrent_kda

    if initial_state is None:
        tail = (v.shape[-1], k.shape[-1]) if state_v_first else (k.shape[-1], v.shape[-1])
        state = q.new_zeros(q.shape[0], v.shape[2], *tail, dtype=torch.float32)
    else:
        state = initial_state if inplace_final_state else initial_state.clone()
    output, _ = recurrent_kda(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        initial_state=state,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        output_final_state=output_final_state,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound,
        state_v_first=state_v_first,
    )
    if out is not None:
        out.copy_(output)
        output = out
    final_state = state if inplace_final_state or output_final_state else None
    return output, final_state
