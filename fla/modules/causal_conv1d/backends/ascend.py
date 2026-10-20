# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.backends import BaseBackend, register
from fla.modules.causal_conv1d.ops import causal_conv1d_bwd, causal_conv1d_fwd, causal_conv1d_update
from fla.utils.ascendc import ensure_ascendc_wheel, is_ascendc_available, verify_ascendc_tensors

ensure_ascendc_wheel()


def _verify_input(x, weight, bias, activation):
    supported, reason = verify_ascendc_tensors(x, weight, bias)
    if not supported:
        return supported, reason
    if weight is None or weight.ndim != 2 or weight.shape[1] not in (2, 3, 4):
        return False, 'AscendC convolution requires weights [D, W], W=2/3/4'
    if x.ndim != 3 or x.shape[-1] != weight.shape[0] or x.shape[-1] % 16:
        return False, 'AscendC convolution requires [B, T, D] with D divisible by 16'
    if x.dtype not in (torch.float16, torch.bfloat16) or weight.dtype != x.dtype:
        return False, 'AscendC convolution requires matching float16/bfloat16 input and weights'
    if bias is not None and (bias.shape != (x.shape[-1],) or bias.dtype != x.dtype):
        return False, 'AscendC convolution bias must match the channels and dtype'
    if activation not in (None, 'silu', 'swish'):
        return False, 'AscendC convolution supports None/silu/swish activation'
    return True, None


@register(causal_conv1d_fwd, env_var='FLA_CONV_ASCENDC')
@register(causal_conv1d_bwd, env_var='FLA_CONV_ASCENDC')
@register(causal_conv1d_update, env_var='FLA_CONV_ASCENDC')
class AscendCConvBackend(BaseBackend):
    backend_type = 'ascendc'
    package_name = 'fla_npu'
    default_enable = False
    priority = -1

    @classmethod
    def is_available(cls):
        return is_ascendc_available('FLA_CONV_ASCENDC')

    def causal_conv1d_fwd_verifier(
        self, x, weight, bias, residual, initial_state=None, output_final_state=False, activation=None,
        cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None, chunk_size=64, layout_fallback=False,
    ):
        supported, reason = _verify_input(x, weight, bias, activation)
        if not supported:
            return supported, reason
        if any(t is not None for t in (cu_seqlens, cu_seqlens_cpu, chunk_indices)):
            return False, 'AscendC convolution currently supports dense inputs'
        if residual is not None:
            if residual.shape != x.shape or residual.dtype != x.dtype or residual.device != x.device:
                return False, 'AscendC convolution residual must match the input'
        if initial_state is not None:
            if initial_state.shape != (x.shape[0], *weight.shape) or initial_state.dtype != x.dtype:
                return False, 'AscendC convolution initial state must be [B, D, W] with input dtype'
            if initial_state.device != x.device:
                return False, 'AscendC convolution states must be on the input device'
        return True, None

    def causal_conv1d_fwd(
        self, x, weight, bias, residual, initial_state=None, output_final_state=False, activation=None,
        cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None, chunk_size=64, layout_fallback=False,
    ):
        from fla_npu.ops.ascendc import causal_conv1d_fn

        width = weight.shape[1]
        state = x.new_zeros(x.shape[0], width - 1, weight.shape[0])
        if initial_state is not None:
            state.copy_(initial_state[:, :, 1:].transpose(1, 2))
        result = causal_conv1d_fn(
            x=x.contiguous(),
            weight=weight.t().contiguous(),
            bias=bias,
            conv_states=state,
            activation=activation,
            has_initial_state_cpu=[int(initial_state is not None)] * x.shape[0],
        )
        if residual is not None:
            result = result + residual
        final_state = None
        if output_final_state:
            prefix = x.new_zeros(x.shape[0], x.shape[2], width) if initial_state is None else initial_state
            final_state = torch.cat((prefix, x.transpose(1, 2)), dim=2)[:, :, -width:].contiguous()
        return result, final_state

    def causal_conv1d_bwd_verifier(
        self, x, dy, dht, weight=None, bias=None, residual=None, initial_state=None, activation=None,
        cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None, chunk_size=64, layout_fallback=False,
    ):
        supported, reason = self.causal_conv1d_fwd_verifier(
            x, weight, bias, residual, initial_state, False, activation,
            cu_seqlens, cu_seqlens_cpu, chunk_indices, chunk_size, layout_fallback,
        )
        if not supported:
            return supported, reason
        if dy.shape != x.shape or dy.dtype != x.dtype or dy.device != x.device:
            return False, 'AscendC convolution gradient must match the input'
        if dht is not None:
            if dht.shape != (x.shape[0], *weight.shape) or dht.dtype != x.dtype or dht.device != x.device:
                return False, 'AscendC final state gradient must match [B, D, W]'
        return True, None

    def causal_conv1d_bwd(
        self, x, dy, dht, weight=None, bias=None, residual=None, initial_state=None, activation=None,
        cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None, chunk_size=64, layout_fallback=False,
    ):
        from fla_npu.ops.ascendc import causal_conv1d_bwd as native_backward

        y = None
        if activation is not None:
            y, _ = self.causal_conv1d_fwd(x, weight, bias, None, initial_state=initial_state)
        dx, dw, db, dh0 = native_backward(
            x=x.contiguous(),
            y=y,
            weight=weight.t().contiguous(),
            dy=dy.contiguous(),
            initial_state=None if initial_state is None else initial_state.transpose(1, 2).contiguous(),
            dht=None if dht is None else dht.transpose(1, 2).contiguous(),
            activation=0 if activation is None else 1,
            input_layout='BSND',
        )
        return (
            dx,
            dw.t().contiguous(),
            db if bias is not None else None,
            dy if residual is not None else None,
            dh0.transpose(1, 2).contiguous() if initial_state is not None else None,
        )

    def causal_conv1d_update_verifier(self, x, cache, residual=None, weight=None, bias=None, activation=None):
        supported, reason = _verify_input(x, weight, bias, activation)
        if not supported:
            return supported, reason
        if x.shape[1] != 1:
            return False, 'AscendC convolution update requires [B, 1, D]'
        if cache.shape != (x.shape[0], *weight.shape) or cache.dtype != x.dtype or cache.device != x.device:
            return False, 'AscendC convolution cache must match [B, D, W]'
        if cache.requires_grad:
            return False, 'AscendC convolution cache must not require gradients'
        if residual is not None and (residual.shape != x.shape or residual.dtype != x.dtype or residual.device != x.device):
            return False, 'AscendC convolution residual must match the input'
        return True, None

    def causal_conv1d_update(self, x, cache, residual=None, weight=None, bias=None, activation=None):
        from fla_npu.ops.ascendc import causal_conv1d_update as native_update

        state = cache[:, :, 1:].transpose(1, 2).contiguous()
        result = native_update(
            x=x.clone(),
            conv_state=state,
            weight=weight.t().contiguous(),
            bias=bias,
            activation=activation,
        )
        cache.copy_(torch.cat((cache[:, :, 1:], x.transpose(1, 2)), dim=2))
        return (result if residual is None else result + residual), cache
