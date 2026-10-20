# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import torch

from fla.backends import BaseBackend, register
from fla.utils.ascendc import ensure_ascendc_wheel, is_ascendc_available, verify_ascendc_tensors

ensure_ascendc_wheel()


@register('gla', env_var='FLA_GLA_ASCENDC')
class AscendCGLABackend(BaseBackend):
    backend_type = 'ascendc'
    package_name = 'fla_npu'
    default_enable = False
    priority = -1

    @classmethod
    def is_available(cls):
        return is_ascendc_available('FLA_GLA_ASCENDC')

    def chunk_gla_fwd_o_gk_verifier(
        self, q, v, g, A, h, scale, state_v_first=False, cu_seqlens=None,
        chunk_size=64, chunk_indices=None, use_graph=False,
    ):
        supported, reason = verify_ascendc_tensors(q, v, g, A, h)
        if not supported:
            return supported, reason
        if cu_seqlens is not None or chunk_indices is not None or use_graph or chunk_size != 64:
            return False, 'AscendC vector-gated output requires dense eager calls and chunk_size=64'
        if q.ndim != 4 or q.shape != v.shape or q.shape != g.shape or q.shape[-1] != 128:
            return False, 'AscendC vector-gated output requires equal heads and K=V=128'
        if any(x.dtype != torch.bfloat16 for x in (q, v, A, h)) or g.dtype != torch.float32:
            return False, 'AscendC vector-gated output requires bfloat16 tensors and float32 gates'
        if A.shape != (*q.shape[:3], 64) or h.shape != (q.shape[0], math.ceil(q.shape[1] / 64), q.shape[2], 128, 128):
            return False, 'AscendC vector-gated output requires matching chunk matrices and states'
        return True, None

    def chunk_gla_fwd_o_gk(
        self, q, v, g, A, h, scale, state_v_first=False, cu_seqlens=None,
        chunk_size=64, chunk_indices=None, use_graph=False,
    ):
        from fla_npu.ops.ascendc import chunk_kda_fwd_finalize

        qg = (q.float() * torch.exp2(g) * scale).to(q.dtype)
        row = torch.arange(q.shape[1], device=q.device) % chunk_size
        col = torch.arange(chunk_size, device=q.device)
        aqk = A.masked_fill(row[None, :, None, None] < col[None, None, None, :], 0)
        return chunk_kda_fwd_finalize(
            qg_scaled=qg.transpose(1, 2).contiguous(),
            aqk=aqk.transpose(1, 2).contiguous(),
            v_new=v.transpose(1, 2).contiguous(),
            h=h.contiguous(),
            output_layout='BSND',
            state_v_first=state_v_first,
        )
