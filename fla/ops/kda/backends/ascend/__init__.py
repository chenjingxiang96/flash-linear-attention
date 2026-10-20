# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Optional Ascend wheel implementations for KDA operations."""

from fla.backends import BaseBackend, register
from fla.utils.ascendc import ensure_ascendc_wheel, is_ascendc_available

ensure_ascendc_wheel()


@register('kda', env_var='FLA_KDA_ASCENDC_OPS')
class AscendCKDAOpsBackend(BaseBackend):
    backend_type = 'ascendc'
    package_name = 'fla_npu'
    default_enable = False
    priority = -1

    @classmethod
    def is_available(cls) -> bool:
        return is_ascendc_available('FLA_KDA_ASCENDC_OPS')

    def kda_gate_chunk_cumsum_verifier(self, *args, **kwargs):
        from fla.ops.kda.backends.ascend.gate import kda_gate_chunk_cumsum_verifier
        return kda_gate_chunk_cumsum_verifier(*args, **kwargs)

    def kda_gate_chunk_cumsum(self, *args, **kwargs):
        from fla.ops.kda.backends.ascend.gate import kda_gate_chunk_cumsum
        return kda_gate_chunk_cumsum(*args, **kwargs)

    def chunk_kda_bwd_intra_verifier(self, *args, **kwargs):
        from fla.ops.kda.backends.ascend.chunk_intra import chunk_kda_bwd_intra_verifier
        return chunk_kda_bwd_intra_verifier(*args, **kwargs)

    def chunk_kda_bwd_intra(self, *args, **kwargs):
        from fla.ops.kda.backends.ascend.chunk_intra import chunk_kda_bwd_intra
        return chunk_kda_bwd_intra(*args, **kwargs)

    def fused_recurrent_kda_fwd_verifier(self, *args, **kwargs):
        from fla.ops.kda.backends.ascend.recurrent import fused_recurrent_kda_fwd_verifier
        return fused_recurrent_kda_fwd_verifier(*args, **kwargs)

    def fused_recurrent_kda_fwd(self, *args, **kwargs):
        from fla.ops.kda.backends.ascend.recurrent import fused_recurrent_kda_fwd
        return fused_recurrent_kda_fwd(*args, **kwargs)

    def chunk_kda_bwd_verifier(self, *args, **kwargs):
        from fla.ops.kda.backends.ascend.chunk_bwd import chunk_kda_bwd_verifier
        return chunk_kda_bwd_verifier(*args, **kwargs)

    def chunk_kda_bwd(self, *args, **kwargs):
        from fla.ops.kda.backends.ascend.chunk_bwd import chunk_kda_bwd
        return chunk_kda_bwd(*args, **kwargs)
