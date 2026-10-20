# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import torch

from fla.utils.ascendc import verify_ascendc_tensors


def chunk_kda_bwd_verifier(
    q, k, v, beta, Aqk, Akk, scale, initial_state, do, dht, g=None, g_org=None,
    state_v_first=False, cu_seqlens=None, chunk_indices=None, chunk_size=64,
    safe_gate=False, lower_bound=None, use_gate_in_kernel=False, A_log=None,
    dt_bias=None, disable_recompute=False, cp_context=None, use_graph=False,
    chunk_offsets=None, **kwargs,
):
    supported, reason = verify_ascendc_tensors(q, k, v, beta, Aqk, Akk, do, g)
    if not supported:
        return supported, reason
    if not disable_recompute or not safe_gate or use_gate_in_kernel or state_v_first or use_graph or chunk_size != 64:
        return False, 'AscendC fused KDA backward requires saved intermediates, safe gates and chunk_size=64'
    if any(x is not None for x in (initial_state, dht, g_org, cu_seqlens, chunk_indices, cp_context, chunk_offsets, A_log, dt_bias)):
        return False, 'AscendC fused KDA backward currently excludes boundary states, varlen and gate activation'
    if q.ndim != 4 or q.shape[-1] != 128 or any(x.shape != q.shape for x in (k, v, do)):
        return False, 'AscendC fused KDA backward requires equal heads and K=V=128'
    if any(x.dtype != torch.bfloat16 for x in (q, k, v, do, Aqk, Akk)):
        return False, 'AscendC fused KDA backward requires bfloat16 data and chunk matrices'
    if g is None or g.shape != q.shape or g.dtype != torch.float32 or beta.shape != q.shape[:3]:
        return False, 'AscendC fused KDA backward requires float32 cumulative gates and scalar beta'
    if beta.dtype not in (torch.bfloat16, torch.float32) or any(x.shape != (*q.shape[:3], 64) for x in (Aqk, Akk)):
        return False, 'AscendC fused KDA backward requires matching beta and chunk matrices'
    expected = {'w': q.shape, 'u': v.shape, 'qg': q.shape, 'kg': k.shape, 'v_new': v.shape,
                'h': (q.shape[0], math.ceil(q.shape[1] / 64), q.shape[2], 128, 128)}
    if kwargs.keys() != expected.keys():
        return False, 'AscendC fused KDA backward requires all saved intermediates'
    for name, shape in expected.items():
        tensor = kwargs[name]
        if tensor is None or tensor.shape != shape or tensor.dtype != q.dtype or tensor.device != q.device:
            return False, 'AscendC fused KDA backward received incompatible saved intermediates'
    return True, None


def chunk_kda_bwd(
    q, k, v, beta, Aqk, Akk, scale, initial_state, do, dht, g=None, g_org=None,
    state_v_first=False, cu_seqlens=None, chunk_indices=None, chunk_size=64,
    safe_gate=False, lower_bound=None, use_gate_in_kernel=False, A_log=None,
    dt_bias=None, disable_recompute=False, cp_context=None, use_graph=False,
    chunk_offsets=None, **kwargs,
):
    from fla_npu.ops.ascendc import chunk_kda_bwd as native_backward

    def head_first(tensor):
        return tensor.transpose(1, 2).contiguous()

    results = native_backward(
        q=head_first(q),
        k=head_first(k),
        v=head_first(v),
        beta=head_first(beta),
        gk=head_first(g),
        Aqk=head_first(Aqk),
        Akk=head_first(Akk),
        w=head_first(kwargs['w']),
        qg=head_first(kwargs['qg']),
        kg=head_first(kwargs['kg']),
        v_new=head_first(kwargs['v_new']),
        h=kwargs['h'].contiguous(),
        d_o=head_first(do),
        scale=scale,
        chunk_size=64,
        safe_gate=True,
        lower_bound=lower_bound,
        disable_recompute=True,
        use_exp2=True,
    )
    return tuple(head_first(x) if i < 5 else x for i, x in enumerate(results))
