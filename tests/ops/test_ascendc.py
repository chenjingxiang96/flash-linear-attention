# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from unittest.mock import patch

import pytest
import torch

from fla.utils import IS_NPU, assert_close

pytestmark = pytest.mark.skipif(not IS_NPU, reason='AscendC requires an NPU')


@pytest.fixture
def native_ops(monkeypatch):
    if 'Ascend950' not in torch.npu.get_device_name():
        pytest.skip('AscendC adapters require A5')
    module = pytest.importorskip('fla_npu.ops.ascendc')
    for flag in (
        'FLA_ASCENDC', 'FLA_UTILS_ASCENDC', 'FLA_COMMON_ASCENDC', 'FLA_GDN_ASCENDC',
        'FLA_GLA_ASCENDC', 'FLA_KDA_ASCENDC', 'FLA_KDA_ASCENDC_OPS', 'FLA_CONV_ASCENDC', 'FLA_CP_ASCENDC',
    ):
        monkeypatch.setenv(flag, '0')
    return module


def compare_dispatch(monkeypatch, native_ops, entry, native_name, reference=None, **kwargs):
    expected = entry(**kwargs) if reference is None else reference
    monkeypatch.setenv('FLA_ASCENDC', '1')
    with patch.object(native_ops, native_name, wraps=getattr(native_ops, native_name)) as native:
        actual = entry(**kwargs)
        assert native.call_count == 1
    expected = expected if isinstance(expected, tuple) else (expected,)
    actual = actual if isinstance(actual, tuple) else (actual,)
    assert len(expected) == len(actual)
    for reference, candidate in zip(expected, actual):
        if reference is None:
            assert candidate is None
        else:
            assert reference.shape == candidate.shape
            assert reference.dtype == candidate.dtype
            assert torch.isfinite(candidate).all()
            assert_close(native_name, reference, candidate, 0.005)


@pytest.mark.parametrize(('operation', 'tokens'),
                         [(op, tokens) for op in ('cumsum', 'solve', 'kkt', 'state', 'output', 'wy',
                                                  'wy_backward', 'dv', 'dqkwg', 'dhu', 'gdn', 'finalize', 'kda', 'kda_gate', 'kda_intra') for tokens in (64, 93)]
                         + [(op, tokens) for op in ('recurrent', 'gdn_recurrent') for tokens in (1, 4, 8)])
@torch.no_grad()
def test_ascendc_dense_dispatch(monkeypatch, native_ops, operation, tokens):
    torch.manual_seed(42)
    shape = (1, tokens, 2, 128)
    k = torch.randn(shape, device='npu', dtype=torch.bfloat16) * 0.05
    v = torch.randn_like(k) * 0.05
    beta = torch.rand(shape[:3], device='npu', dtype=torch.float32)
    gate = -torch.rand(shape[:3], device='npu', dtype=torch.float32) * 0.01
    for offset in range(0, tokens, 64):
        gate[:, offset:offset + 64] = gate[:, offset:offset + 64].cumsum(1)
    if operation == 'cumsum':
        from fla.ops.utils import chunk_local_cumsum
        compare_dispatch(
            monkeypatch,
            native_ops,
            chunk_local_cumsum,
            'chunk_local_cumsum',
            g=gate,
            chunk_size=64,
            reverse=True,
            scale=0.5,
        )
    elif operation == 'solve':
        from fla.ops.utils import solve_tril
        A = torch.randn(1, tokens, 2, 64, device='npu', dtype=torch.bfloat16) * 0.01
        row = torch.arange(tokens, device='npu') % 64
        col = torch.arange(64, device='npu')
        A.masked_fill_(row[None, :, None, None] <= col[None, None, None, :], 0)
        compare_dispatch(monkeypatch, native_ops, solve_tril, 'solve_tri', A=A, output_dtype=A.dtype)
    elif operation == 'kkt':
        from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
        compare_dispatch(
            monkeypatch,
            native_ops,
            chunk_scaled_dot_kkt_fwd,
            'chunk_scaled_dot_kkt',
            k=k,
            beta=beta,
            g=gate,
        )
    elif operation == 'state':
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
        compare_dispatch(
            monkeypatch,
            native_ops,
            chunk_gated_delta_rule_fwd_h,
            'chunk_fwd_h',
            k=k,
            w=k,
            u=v,
            g=gate,
            output_final_state=True,
        )
    elif operation == 'output':
        from fla.ops.common.chunk_o import chunk_fwd_o
        h = torch.randn(1, (tokens + 63) // 64, 2, 128, 128, device='npu', dtype=k.dtype) * 0.05
        compare_dispatch(monkeypatch, native_ops, chunk_fwd_o, 'chunk_fwd_o', q=k, k=k, v=v, h=h, g=gate)
    elif operation == 'finalize':
        from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
        h = torch.randn(1, (tokens + 63) // 64, 2, 128, 128, device='npu', dtype=k.dtype) * 0.05
        A = torch.randn(1, tokens, 2, 64, device='npu', dtype=k.dtype) * 0.01
        gate = gate[..., None].expand_as(k).contiguous()
        compare_dispatch(
            monkeypatch,
            native_ops,
            chunk_gla_fwd_o_gk,
            'chunk_kda_fwd_finalize',
            q=k,
            v=v,
            g=gate,
            A=A,
            h=h,
            scale=128 ** -0.5,
        )
    elif operation == 'dv':
        from fla.ops.common.chunk_o import chunk_bwd_dv_local
        compare_dispatch(
            monkeypatch,
            native_ops,
            chunk_bwd_dv_local,
            'chunk_bwd_dv_local',
            q=k,
            k=k,
            do=v,
            g=gate,
            scale=128 ** -0.5,
        )
    elif operation == 'dqkwg':
        from fla.ops.common.chunk_o import chunk_bwd_dqkwg
        h = torch.randn(1, (tokens + 63) // 64, 2, 128, 128, device='npu', dtype=k.dtype) * 0.05
        compare_dispatch(
            monkeypatch,
            native_ops,
            chunk_bwd_dqkwg,
            'chunk_bwd_dqkwg',
            q=k,
            k=k,
            v=v,
            do=v,
            h=h,
            dh=h,
            w=k,
            g=gate,
            dv=v,
            scale=128 ** -0.5,
        )
    elif operation == 'dhu':
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
        compare_dispatch(
            monkeypatch,
            native_ops,
            chunk_gated_delta_rule_bwd_dhu,
            'chunk_gated_delta_rule_bwd_dhu',
            q=k,
            k=k,
            w=k,
            do=v,
            dv=v,
            g=gate,
            scale=128 ** -0.5,
        )
    elif operation == 'gdn_recurrent':
        from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd
        from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
        beta = beta.to(k.dtype)
        reference = naive_recurrent_gated_delta_rule(
            k.cpu(),
            k.cpu(),
            v.cpu(),
            beta=beta.cpu(),
            g=gate.cpu(),
            output_final_state=True,
        )
        reference = (reference[0].to(k), reference[1].to(k.device))
        compare_dispatch(
            monkeypatch,
            native_ops,
            fused_recurrent_gated_delta_rule_fwd,
            'recurrent_gated_delta_rule',
            q=k,
            k=k,
            v=v,
            g=gate,
            beta=beta,
            scale=128 ** -0.5,
            output_final_state=True,
            reference=reference,
        )
    elif operation == 'kda':
        from fla.ops.kda import chunk_kda
        gate = -torch.rand(shape, device='npu') * 0.01
        compare_dispatch(
            monkeypatch,
            native_ops,
            chunk_kda,
            'chunk_kda_fwd',
            q=k,
            k=k,
            v=v,
            g=gate,
            beta=beta,
            output_final_state=True,
        )
    elif operation == 'kda_gate':
        from fla.ops.kda.gate import kda_gate_chunk_cumsum
        from fla.ops.utils.constant import RCP_LN2
        raw_gate = torch.randn(shape, device='npu', dtype=k.dtype)
        A_log = torch.randn(shape[2], device='npu')
        dt_bias = torch.randn(shape[2] * shape[3], device='npu')
        compare_dispatch(
            monkeypatch,
            native_ops,
            kda_gate_chunk_cumsum,
            'kda_gate_cumsum',
            g=raw_gate,
            A_log=A_log,
            dt_bias=dt_bias,
            chunk_size=64,
            scale=RCP_LN2,
            lower_bound=-5.0,
        )
    elif operation == 'kda_intra':
        from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
        gate = gate[..., None].expand_as(k).contiguous()
        dA = torch.randn(*shape[:3], 64, device='npu') * 0.01
        kwargs = dict(
            q=k,
            k=k,
            g=gate,
            beta=beta,
            dAqk=dA,
            dAkk=dA,
            dq=torch.zeros_like(gate),
            dk=torch.zeros_like(gate),
            db=torch.zeros_like(beta),
            dg=torch.zeros_like(gate),
            safe_gate=True,
        )
        reference = chunk_kda_bwd_intra(**{name: value.clone() if isinstance(value, torch.Tensor) else value
                                         for name, value in kwargs.items()})
        compare_dispatch(
            monkeypatch,
            native_ops,
            chunk_kda_bwd_intra,
            'chunk_kda_bwd_intra',
            reference=reference,
            **kwargs,
        )
    elif operation == 'gdn':
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        compare_dispatch(
            monkeypatch,
            native_ops,
            chunk_gated_delta_rule,
            'chunk_gated_delta_rule_fwd',
            q=k,
            k=k,
            v=v,
            g=gate,
            beta=beta,
            output_final_state=True,
        )
    elif operation == 'recurrent':
        from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd
        from fla.ops.kda.naive import naive_recurrent_kda
        gate = -torch.rand(shape, device='npu', dtype=torch.float32) * 0.01
        reference = naive_recurrent_kda(k.cpu(), k.cpu(), v.cpu(), gate.cpu(), beta.cpu(), output_final_state=True)
        reference = tuple(x.to(k.device) for x in reference)
        compare_dispatch(
            monkeypatch,
            native_ops,
            fused_recurrent_kda_fwd,
            'recurrent_kda',
            q=k,
            k=k,
            v=v,
            g=gate,
            beta=beta,
            inplace_final_state=False,
            output_final_state=True,
            reference=reference,
        )
    else:
        from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
        A = torch.randn(1, tokens, 2, 64, device='npu', dtype=k.dtype) * 0.01
        row = torch.arange(tokens, device='npu') % 64
        col = torch.arange(64, device='npu')
        A.masked_fill_(row[None, :, None, None] < col[None, None, None, :], 0)
        A += (row[None, :, None, None] == col[None, None, None, :]).to(A.dtype)
        entry = recompute_w_u_fwd if operation == 'wy' else prepare_wy_repr_bwd
        extra = {} if operation == 'wy' else {'dw': k, 'du': v}
        reference = None
        if operation == 'wy_backward':
            A, reference = wy_backward_reference(k, v, beta, gate, extra['dw'], extra['du'])
        compare_dispatch(
            monkeypatch,
            native_ops,
            entry,
            entry.__name__,
            reference=reference,
            k=k,
            v=v,
            beta=beta,
            A=A,
            g=gate,
            **extra,
        )


@torch.no_grad()
def test_ascendc_cumsum_fallback_and_error(monkeypatch, native_ops):
    from fla.ops.utils import chunk_local_cumsum
    monkeypatch.setenv('FLA_ASCENDC', '1')
    g = torch.randn(1, 128, 2, device='npu')
    cu = torch.tensor([0, 64, 128], device='npu', dtype=torch.int32)
    with patch.object(native_ops, 'chunk_local_cumsum', side_effect=RuntimeError('native failure')) as native:
        result = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu)
        assert native.call_count == 0
        assert torch.isfinite(result).all()
        with pytest.raises(RuntimeError, match='native failure'):
            chunk_local_cumsum(g, chunk_size=64)
        assert native.call_count == 1


@pytest.mark.parametrize('activation', [None, 'silu'])
@pytest.mark.parametrize('with_state', [False, True])
@torch.no_grad()
def test_ascendc_conv(monkeypatch, native_ops, activation, with_state):
    from fla.modules.causal_conv1d.ops import causal_conv1d_bwd, causal_conv1d_fwd, causal_conv1d_update
    torch.manual_seed(42)
    x = torch.randn(2, 17, 32, device='npu', dtype=torch.bfloat16) * 0.1
    weight = torch.randn(32, 4, device='npu', dtype=x.dtype) * 0.1
    bias = torch.randn(32, device='npu', dtype=x.dtype) * 0.1
    state = torch.randn(2, 32, 4, device='npu', dtype=x.dtype) * 0.1 if with_state else None
    saved_state = None if state is None else state.clone()
    compare_dispatch(
        monkeypatch,
        native_ops,
        causal_conv1d_fwd,
        'causal_conv1d_fn',
        x=x,
        weight=weight,
        bias=bias,
        residual=None,
        initial_state=state,
        output_final_state=True,
        activation=activation,
    )
    if state is not None:
        torch.testing.assert_close(saved_state, state, rtol=0, atol=0)
    monkeypatch.setenv('FLA_ASCENDC', '0')
    compare_dispatch(
        monkeypatch,
        native_ops,
        causal_conv1d_bwd,
        'causal_conv1d_bwd',
        x=x,
        dy=x,
        dht=None,
        weight=weight,
        bias=bias,
        initial_state=state,
        activation=activation,
    )
    cache = torch.zeros(2, 32, 4, device='npu', dtype=x.dtype)
    expected_cache = cache.clone()
    token = x[:, :1].contiguous()
    saved_token = token.clone()
    monkeypatch.setenv('FLA_ASCENDC', '0')
    expected = causal_conv1d_update(token, expected_cache, weight=weight, bias=bias, activation=activation)
    compare_dispatch(
        monkeypatch,
        native_ops,
        causal_conv1d_update,
        'causal_conv1d_update',
        reference=expected,
        x=token,
        cache=cache,
        weight=weight,
        bias=bias,
        activation=activation,
    )
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
    torch.testing.assert_close(token, saved_token, rtol=0, atol=0)


@torch.enable_grad()
def wy_backward_reference(k, v, beta, gate, dw, du):
    import math
    keys, values, betas = [x.detach().float().cpu().requires_grad_() for x in (k, v, beta)]
    gates = (gate.detach().float().cpu() * math.log(2)).requires_grad_()
    A = torch.zeros(*k.shape[:3], 64)
    loss = 0
    for start in range(0, k.shape[1], 64):
        end = min(start + 64, k.shape[1])
        kc = keys[:, start:end].transpose(1, 2)
        vc = values[:, start:end].transpose(1, 2)
        bc = betas[:, start:end].transpose(1, 2)
        gc = gates[:, start:end].transpose(1, 2)
        lower = ((kc @ kc.transpose(-1, -2)) * bc[..., None] * (gc[..., None] - gc[..., None, :]).exp()).tril(-1)
        inverse = torch.linalg.inv(torch.eye(end - start) + lower)
        A[:, start:end, :, :end - start] = inverse.detach().transpose(1, 2)
        w = inverse @ (kc * bc[..., None] * gc.exp()[..., None])
        u = inverse @ (vc * bc[..., None])
        loss = loss + (w * dw[:, start:end].float().cpu().transpose(1, 2)).sum()
        loss = loss + (u * du[:, start:end].float().cpu().transpose(1, 2)).sum()
    gradients = torch.autograd.grad(loss, (keys, values, betas, gates))
    return A.to(k), tuple(grad.to(original) for grad, original in zip(gradients, (k, v, beta, gate)))


@pytest.mark.parametrize('tokens', [64, 93])
def test_ascendc_gdn_training(monkeypatch, native_ops, tokens):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule

    torch.manual_seed(42)
    shape = (1, tokens, 2, 128)
    inputs = [torch.randn(shape, device='npu', dtype=torch.bfloat16) * 0.05 for _ in range(3)]
    inputs += [torch.rand(shape[:3], device='npu'), -torch.rand(shape[:3], device='npu') * 0.01]
    reference_inputs = [x.detach().cpu().requires_grad_() for x in inputs]
    inputs = [x.requires_grad_() for x in inputs]
    do = torch.randn(shape, device='npu', dtype=torch.bfloat16)
    expected, _ = naive_recurrent_gated_delta_rule(*reference_inputs)
    expected.backward(do.cpu())
    monkeypatch.setenv('FLA_ASCENDC', '1')
    with patch.object(native_ops, 'chunk_gated_delta_rule_bwd', wraps=native_ops.chunk_gated_delta_rule_bwd) as native:
        actual, _ = chunk_gated_delta_rule(q=inputs[0], k=inputs[1], v=inputs[2], beta=inputs[3], g=inputs[4])
        actual.backward(do)
        assert native.call_count == 1
    assert_close('GDN output', expected.to(actual), actual, 0.005)
    for name, reference, candidate in zip(('q', 'k', 'v', 'beta', 'g'), reference_inputs, inputs):
        assert torch.isfinite(candidate.grad).all()
        assert_close(f'GDN d{name}', reference.grad.to(candidate.grad),
                     candidate.grad, 0.01 if name in ('beta', 'g') else 0.005)


@pytest.mark.parametrize('operation', ['kda', 'gdn'])
@torch.no_grad()
def test_ascendc_recurrent_token_limit(native_ops, operation):
    from fla.ops.gated_delta_rule.backends.ascend.ops import fused_recurrent_gated_delta_rule_fwd_verifier
    from fla.ops.kda.backends.ascend.recurrent import fused_recurrent_kda_fwd_verifier

    q = torch.randn(1, 9, 2, 128, device='npu', dtype=torch.bfloat16)
    beta = torch.rand(q.shape[:3], device='npu', dtype=q.dtype)
    gate = torch.zeros(q.shape if operation == 'kda' else q.shape[:3], device='npu')
    verifier = fused_recurrent_kda_fwd_verifier if operation == 'kda' else fused_recurrent_gated_delta_rule_fwd_verifier
    supported, reason = verifier(q=q, k=q, v=q, g=gate, beta=beta)
    assert not supported
    assert '8 tokens' in reason


@pytest.mark.parametrize('tokens', [64, 93])
def test_ascendc_kda_training(monkeypatch, native_ops, tokens):
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.naive import naive_recurrent_kda

    torch.manual_seed(42)
    shape = (1, tokens, 2, 128)
    inputs = [torch.randn(shape, device='npu', dtype=torch.bfloat16) * 0.05 for _ in range(3)]
    inputs += [-torch.rand(shape, device='npu') * 0.01, torch.rand(shape[:3], device='npu')]
    reference_inputs = [x.detach().cpu().requires_grad_() for x in inputs]
    inputs = [x.requires_grad_() for x in inputs]
    do = torch.randn(shape, device='npu', dtype=torch.bfloat16)
    expected, _ = naive_recurrent_kda(*reference_inputs)
    expected.backward(do.cpu())
    monkeypatch.setenv('FLA_ASCENDC', '1')
    with patch.object(native_ops, 'chunk_kda_bwd', wraps=native_ops.chunk_kda_bwd) as native:
        actual, _ = chunk_kda(*inputs, safe_gate=True, disable_recompute=True)
        actual.backward(do)
        assert native.call_count == 1
    assert_close('KDA output', expected.to(actual), actual, 0.005)
    for name, reference, candidate in zip(('q', 'k', 'v', 'g', 'beta'), reference_inputs, inputs):
        assert torch.isfinite(candidate.grad).all()
        assert_close(f'KDA d{name}', reference.grad.to(candidate.grad),
                     candidate.grad, 0.01 if name in ('beta', 'g') else 0.005)


@pytest.mark.parametrize('operation', ['kda', 'gdn'])
@pytest.mark.parametrize('tokens', [1, 8])
@pytest.mark.parametrize('state_v_first', [False, True])
@torch.no_grad()
def test_ascendc_recurrent_states(monkeypatch, native_ops, operation, tokens, state_v_first):
    from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
    from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd
    from fla.ops.kda.naive import naive_recurrent_kda

    torch.manual_seed(42)
    key_heads, value_dim = (1, 256) if operation == 'kda' else (2, 128)
    q = torch.randn(2, tokens, key_heads, 128, device='npu', dtype=torch.bfloat16) * 0.05
    k = torch.randn_like(q) * 0.05
    v = torch.randn(2, tokens, 2, value_dim, device='npu', dtype=q.dtype) * 0.05
    beta = torch.rand(v.shape[:3], device='npu', dtype=q.dtype)
    gate = -torch.rand((*v.shape[:3], 128) if operation == 'kda' else v.shape[:3], device='npu') * 0.01
    state = torch.randn(2, 2, 128, value_dim, device='npu') * 0.05
    reference_fn = naive_recurrent_kda if operation == 'kda' else naive_recurrent_gated_delta_rule
    expected = reference_fn(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        g=gate.cpu(),
        beta=beta.cpu(),
        initial_state=state.cpu(),
        output_final_state=True,
    )
    if state_v_first:
        state = state.transpose(-1, -2).contiguous()
        expected = (expected[0], expected[1].transpose(-1, -2).contiguous())
    expected = (expected[0].to(v), expected[1].to(state))
    saved_state = state.clone()
    entry = fused_recurrent_kda_fwd if operation == 'kda' else fused_recurrent_gated_delta_rule_fwd
    name = 'recurrent_kda' if operation == 'kda' else 'recurrent_gated_delta_rule'
    extra = {'inplace_final_state': False} if operation == 'kda' else {}
    compare_dispatch(
        monkeypatch,
        native_ops,
        entry,
        name,
        reference=expected,
        q=q,
        k=k,
        v=v,
        g=gate,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        state_v_first=state_v_first,
        scale=128 ** -0.5,
        **extra,
    )
    torch.testing.assert_close(state, saved_state, rtol=0, atol=0)


def cp_affine_reference(k, w, u, q, dv, gate, backward, vector_gate):
    k, w, u, q, dv, gate = [x.cpu()[0].transpose(0, 1).float() for x in (k, w, u, q, dv, gate)]
    state = torch.zeros(k.shape[0], 128, 128)
    chain = torch.eye(128).expand(k.shape[0], 128, 128).clone()
    offsets = list(range(0, k.shape[1], 64))
    for offset in reversed(offsets) if backward else offsets:
        kc, wc, uc, qc, dc, gc = [x[:, offset:offset + 64] for x in (k, w, u, q, dv, gate)]
        decay = gc[:, -1].exp2()
        if not vector_gate:
            decay = decay[:, None].expand(-1, 128)
        factor = (gc[:, -1:] - gc).exp2().unsqueeze(-1) if not vector_gate else 1
        rounded_k = (kc * factor).bfloat16().float()
        if backward:
            projected = (kc @ state.bfloat16().float()) * factor + dc
            qbar = qc * gc.exp2().unsqueeze(-1) if not vector_gate else qc
            state = (decay[:, :, None] * state + qbar.transpose(-1, -2) @ uc * (128 ** -0.5)
                     - wc.transpose(-1, -2) @ projected.bfloat16().float())
            transition = torch.diag_embed(decay) - wc.transpose(-1, -2) @ rounded_k
        else:
            residual = (uc - wc @ state.bfloat16().float()) * factor
            state = decay[:, :, None] * state + kc.transpose(-1, -2) @ residual.bfloat16().float()
            transition = torch.diag_embed(decay) - rounded_k.transpose(-1, -2) @ wc
        chain = transition @ chain
    return torch.cat((state, chain), dim=-1)


@pytest.mark.parametrize('backward', [False, True])
@pytest.mark.parametrize('vector_gate', [False, True])
@pytest.mark.parametrize('tokens', [64, 93, 1024])
@pytest.mark.parametrize('offset', [0, 7])
@torch.no_grad()
def test_ascendc_cp_pre_process(monkeypatch, native_ops, backward, vector_gate, tokens, offset):
    from fla.ops.cp.chunk_delta_h import _pre_process_bwd, _pre_process_fwd

    torch.manual_seed(42)
    shape = (1, tokens + offset, 2, 128)
    k, w, u, q, dv = [torch.randn(shape, device='npu', dtype=torch.bfloat16) * 0.05 for _ in range(5)]
    gate = -torch.rand(shape if vector_gate else shape[:3], device='npu') * 0.01
    for chunk_start in range(0, shape[1], 64):
        gate[:, chunk_start:chunk_start + 64] = gate[:, chunk_start:chunk_start + 64].cumsum(1)
    if backward and vector_gate:
        gate = gate.bfloat16()
    output = torch.empty(2, 128, 256, device='npu', dtype=torch.float32)
    kwargs = dict(
        grid=(4, 2),
        k=k,
        w=w,
        g=None if vector_gate else gate,
        gk=gate if vector_gate else None,
        cu_seqlens=torch.tensor([offset, offset + tokens], device='npu', dtype=torch.int64),
        T=shape[1],
        H=2,
        HV=2,
        K=128,
        V=128,
        BT=64,
        BK1=128,
        BLOCK_SIZE=64,
        AFFINE_CHAIN_PRECISION='ieee',
    )
    if backward:
        entry, name = _pre_process_bwd, 'pre_process_bwd_kernel_merged'
        kwargs.update(q=q, do=u, dv=dv, dhm=output, scale=128 ** -0.5, USE_BG=False)
    else:
        entry, name = _pre_process_fwd, 'pre_process_fwd_kernel_merged'
        kwargs.update(u=u, v=u, bg=None, hm=output, MULTI_SEQS=False)
    reference = cp_affine_reference(
        *(x[:, offset:] for x in (k, w, u, q, dv, gate)), backward, vector_gate,
    ).to(output.device)
    output.fill_(float('nan'))
    monkeypatch.setenv('FLA_ASCENDC', '1')
    with patch.object(native_ops, name, wraps=getattr(native_ops, name)) as native:
        entry(**kwargs)
        assert native.call_count == 1
    assert torch.isfinite(output).all()
    assert_close(name + '_state', reference[..., :128], output[..., :128], 0.005)
    assert_close(name + '_transition', reference[..., 128:], output[..., 128:], 0.005)


@pytest.mark.parametrize('forward', [False, True])
@pytest.mark.parametrize('state_v_first', [False, True])
@torch.no_grad()
def test_ascendc_cp_merge(monkeypatch, native_ops, forward, state_v_first):
    from fla.ops.cp.chunk_delta_h import _merge_states

    torch.manual_seed(42)
    summaries = torch.randn(4, 2, 128, 256, device='npu', dtype=torch.float32) * 0.05
    output = torch.empty(2, 128, 128, device='npu', dtype=torch.float32)
    kwargs = dict(
        grid=lambda meta: (128 // meta['BV'], 2),
        h=output,
        ag_hm=summaries,
        pre_or_post_num_ranks=3,
        rank=3 if forward else 0,
        seq_offsets=None,
        init_offsets=None,
        h0_seq_ids=None,
        h0=None,
        h_seq_idx=None,
        HV=2,
        K=128,
        V=128,
        BK=128,
        FORWARD=forward,
        INTRACARD_MODE=False,
        NUM_SEQ_ENTRIES=0,
        STATE_V_FIRST=state_v_first,
        AFFINE_CHAIN_PRECISION='ieee',
        NUM_RANKS_ON_DEVICE=False,
    )
    reference = torch.zeros(2, 128, 128)
    host_summaries = summaries.cpu()
    for index in (range(3) if forward else range(3, 0, -1)):
        reference = host_summaries[index, ..., 128:] @ reference + host_summaries[index, ..., :128]
    if state_v_first:
        reference = reference.transpose(-1, -2)
    reference = reference.to(output.device)
    output.fill_(float('nan'))
    monkeypatch.setenv('FLA_ASCENDC', '1')
    with patch.object(native_ops, 'merge_fwd_bwd_kernel', wraps=native_ops.merge_fwd_bwd_kernel) as native:
        _merge_states(**kwargs)
        assert native.call_count == 1
    assert torch.isfinite(output).all()
    assert_close('merge_fwd_bwd_kernel', reference, output, 0.005)


@pytest.mark.parametrize('unsupported', ['graph', 'float32_vector_gate', 'disabled'])
@torch.no_grad()
def test_ascendc_cp_fallback(monkeypatch, native_ops, unsupported):
    from fla.ops.cp import chunk_delta_h

    k = torch.zeros(1, 64, 1, 128, device='npu', dtype=torch.bfloat16)
    gate = torch.zeros_like(k, dtype=torch.float32 if unsupported == 'float32_vector_gate' else torch.bfloat16)
    monkeypatch.setenv('FLA_ASCENDC', '0' if unsupported == 'disabled' else '1')
    with patch.object(chunk_delta_h, 'pre_process_bwd_kernel_merged') as fallback, \
            patch.object(native_ops, 'pre_process_bwd_kernel_merged') as native:
        chunk_delta_h._pre_process_bwd(
            grid=(4, 1),
            use_graph=unsupported == 'graph',
            q=k,
            k=k,
            w=k,
            do=k,
            dv=k,
            g=None,
            gk=gate,
            dhm=torch.empty(1, 128, 256, device='npu'),
            cu_seqlens=torch.tensor([0, 64], device='npu'),
            scale=1,
            BT=64,
            USE_BG=False,
            AFFINE_CHAIN_PRECISION='ieee',
        )
        native.assert_not_called()
        fallback.__getitem__.return_value.assert_called_once()


@pytest.mark.parametrize('state_v_first', [False, True])
@pytest.mark.parametrize('output_final_state', [False, True])
@torch.no_grad()
def test_ascendc_gdn_boundary_state(monkeypatch, native_ops, state_v_first, output_final_state):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule

    torch.manual_seed(42)
    q, k, v = [torch.randn(1, 93, 2, 128, device='npu', dtype=torch.bfloat16) * 0.05 for _ in range(3)]
    g = -torch.rand(1, 93, 2, device='npu', dtype=torch.float32) * 0.01
    beta = torch.rand_like(g)
    h0 = torch.randn(1, 2, 128, 128, device='npu', dtype=torch.float32) * 0.05
    original = h0.clone()
    reference = naive_recurrent_gated_delta_rule(
        q=q.cpu().float(),
        k=k.cpu().float(),
        v=v.cpu().float(),
        g=g.cpu(),
        beta=beta.cpu(),
        initial_state=(h0.transpose(-1, -2) if state_v_first else h0).cpu(),
        output_final_state=True,
    )
    expected = (reference[0].to(v), (reference[1].transpose(-1, -2) if state_v_first else reference[1]).to(h0)
                if output_final_state else None)
    compare_dispatch(
        monkeypatch,
        native_ops,
        chunk_gated_delta_rule,
        'chunk_gated_delta_rule_fwd',
        reference=expected,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=output_final_state,
        state_v_first=state_v_first,
    )
    torch.testing.assert_close(h0, original, rtol=0, atol=0)
