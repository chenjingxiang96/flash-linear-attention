# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.backends import BaseBackend, register
from fla.utils.ascendc import ensure_ascendc_wheel, is_ascendc_available

ensure_ascendc_wheel()


@register('utils', env_var='FLA_UTILS_ASCENDC')
class AscendCUtilsBackend(BaseBackend):
    backend_type = 'ascendc'
    package_name = 'fla_npu'
    default_enable = False
    priority = -1

    @classmethod
    def is_available(cls) -> bool:
        return is_ascendc_available('FLA_UTILS_ASCENDC')

    def chunk_local_cumsum_verifier(self, *args, **kwargs):
        from fla.ops.utils.backends.ascend.ops import chunk_local_cumsum_verifier
        return chunk_local_cumsum_verifier(*args, **kwargs)

    def chunk_local_cumsum(self, *args, **kwargs):
        from fla.ops.utils.backends.ascend.ops import chunk_local_cumsum
        return chunk_local_cumsum(*args, **kwargs)

    def solve_tril_verifier(self, *args, **kwargs):
        from fla.ops.utils.backends.ascend.ops import solve_tril_verifier
        return solve_tril_verifier(*args, **kwargs)

    def solve_tril(self, *args, **kwargs):
        from fla.ops.utils.backends.ascend.ops import solve_tril
        return solve_tril(*args, **kwargs)
