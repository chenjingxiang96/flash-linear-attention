# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import BaseBackend, register
from fla.utils.ascendc import ensure_ascendc_wheel, is_ascendc_available

ensure_ascendc_wheel()


@register('gated_delta_rule', env_var='FLA_GDN_ASCENDC')
class AscendCGDNBackend(BaseBackend):
    backend_type = 'ascendc'
    package_name = 'fla_npu'
    default_enable = False
    priority = -1

    @classmethod
    def is_available(cls) -> bool:
        return is_ascendc_available('FLA_GDN_ASCENDC')

    def recompute_w_u_fwd_verifier(self, *args, **kwargs):
        from fla.ops.gated_delta_rule.backends.ascend.ops import recompute_w_u_fwd_verifier
        return recompute_w_u_fwd_verifier(*args, **kwargs)

    def recompute_w_u_fwd(self, *args, **kwargs):
        from fla.ops.gated_delta_rule.backends.ascend.ops import recompute_w_u_fwd
        return recompute_w_u_fwd(*args, **kwargs)

    def prepare_wy_repr_bwd_verifier(self, *args, **kwargs):
        from fla.ops.gated_delta_rule.backends.ascend.ops import prepare_wy_repr_bwd_verifier
        return prepare_wy_repr_bwd_verifier(*args, **kwargs)

    def prepare_wy_repr_bwd(self, *args, **kwargs):
        from fla.ops.gated_delta_rule.backends.ascend.ops import prepare_wy_repr_bwd
        return prepare_wy_repr_bwd(*args, **kwargs)

    def chunk_gated_delta_rule_verifier(self, *args, **kwargs):
        from fla.ops.gated_delta_rule.backends.ascend.ops import chunk_gated_delta_rule_verifier
        return chunk_gated_delta_rule_verifier(*args, **kwargs)

    def chunk_gated_delta_rule(self, *args, **kwargs):
        from fla.ops.gated_delta_rule.backends.ascend.ops import chunk_gated_delta_rule
        return chunk_gated_delta_rule(*args, **kwargs)

    def fused_recurrent_gated_delta_rule_fwd_verifier(self, *args, **kwargs):
        from fla.ops.gated_delta_rule.backends.ascend.ops import fused_recurrent_gated_delta_rule_fwd_verifier
        return fused_recurrent_gated_delta_rule_fwd_verifier(*args, **kwargs)

    def fused_recurrent_gated_delta_rule_fwd(self, *args, **kwargs):
        from fla.ops.gated_delta_rule.backends.ascend.ops import fused_recurrent_gated_delta_rule_fwd
        return fused_recurrent_gated_delta_rule_fwd(*args, **kwargs)

    def chunk_gated_delta_rule_bwd_verifier(self, *args, **kwargs):
        from fla.ops.gated_delta_rule.backends.ascend.ops import chunk_gated_delta_rule_bwd_verifier
        return chunk_gated_delta_rule_bwd_verifier(*args, **kwargs)

    def chunk_gated_delta_rule_bwd(self, *args, **kwargs):
        from fla.ops.gated_delta_rule.backends.ascend.ops import chunk_gated_delta_rule_bwd
        return chunk_gated_delta_rule_bwd(*args, **kwargs)
