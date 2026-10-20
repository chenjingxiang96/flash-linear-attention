# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import BaseBackend, register
from fla.utils.ascendc import ensure_ascendc_wheel, is_ascendc_available

ensure_ascendc_wheel()


@register('common', env_var='FLA_COMMON_ASCENDC')
class AscendCCommonBackend(BaseBackend):
    backend_type = 'ascendc'
    package_name = 'fla_npu'
    default_enable = False
    priority = -1

    @classmethod
    def is_available(cls) -> bool:
        return is_ascendc_available('FLA_COMMON_ASCENDC')

    def chunk_scaled_dot_kkt_fwd_verifier(self, *args, **kwargs):
        from fla.ops.common.backends.ascend.ops import chunk_scaled_dot_kkt_fwd_verifier
        return chunk_scaled_dot_kkt_fwd_verifier(*args, **kwargs)

    def chunk_scaled_dot_kkt_fwd(self, *args, **kwargs):
        from fla.ops.common.backends.ascend.ops import chunk_scaled_dot_kkt_fwd
        return chunk_scaled_dot_kkt_fwd(*args, **kwargs)

    def chunk_gated_delta_rule_fwd_h_verifier(self, *args, **kwargs):
        from fla.ops.common.backends.ascend.ops import chunk_gated_delta_rule_fwd_h_verifier
        return chunk_gated_delta_rule_fwd_h_verifier(*args, **kwargs)

    def chunk_gated_delta_rule_fwd_h(self, *args, **kwargs):
        from fla.ops.common.backends.ascend.ops import chunk_gated_delta_rule_fwd_h
        return chunk_gated_delta_rule_fwd_h(*args, **kwargs)

    def chunk_fwd_o_verifier(self, *args, **kwargs):
        from fla.ops.common.backends.ascend.ops import chunk_fwd_o_verifier
        return chunk_fwd_o_verifier(*args, **kwargs)

    def chunk_fwd_o(self, *args, **kwargs):
        from fla.ops.common.backends.ascend.ops import chunk_fwd_o
        return chunk_fwd_o(*args, **kwargs)

    def chunk_bwd_dv_local_verifier(self, *args, **kwargs):
        from fla.ops.common.backends.ascend.ops import chunk_bwd_dv_local_verifier
        return chunk_bwd_dv_local_verifier(*args, **kwargs)

    def chunk_bwd_dv_local(self, *args, **kwargs):
        from fla.ops.common.backends.ascend.ops import chunk_bwd_dv_local
        return chunk_bwd_dv_local(*args, **kwargs)

    def chunk_bwd_dqkwg_verifier(self, *args, **kwargs):
        from fla.ops.common.backends.ascend.ops import chunk_bwd_dqkwg_verifier
        return chunk_bwd_dqkwg_verifier(*args, **kwargs)

    def chunk_bwd_dqkwg(self, *args, **kwargs):
        from fla.ops.common.backends.ascend.ops import chunk_bwd_dqkwg
        return chunk_bwd_dqkwg(*args, **kwargs)

    def chunk_gated_delta_rule_bwd_dhu_verifier(self, *args, **kwargs):
        from fla.ops.common.backends.ascend.ops import chunk_gated_delta_rule_bwd_dhu_verifier
        return chunk_gated_delta_rule_bwd_dhu_verifier(*args, **kwargs)

    def chunk_gated_delta_rule_bwd_dhu(self, *args, **kwargs):
        from fla.ops.common.backends.ascend.ops import chunk_gated_delta_rule_bwd_dhu
        return chunk_gated_delta_rule_bwd_dhu(*args, **kwargs)
