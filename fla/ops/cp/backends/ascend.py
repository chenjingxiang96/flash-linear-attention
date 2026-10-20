# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.backends import BaseBackend, register
from fla.utils.ascendc import ensure_ascendc_wheel, is_ascendc_available, verify_ascendc_tensors

ensure_ascendc_wheel()


def _verify_pre_process(use_graph, kwargs, backward):
    names = ('q', 'k', 'w', 'do', 'dv') if backward else ('k', 'w', 'u')
    tensors = [kwargs[name] for name in names]
    g, gk = kwargs['g'], kwargs['gk']
    output = kwargs['dhm' if backward else 'hm']
    supported, reason = verify_ascendc_tensors(*tensors, g, gk, output)
    if not supported:
        return supported, reason
    if use_graph or kwargs['BT'] != 64 or kwargs.get('AFFINE_CHAIN_PRECISION') != 'ieee':
        return False, 'AscendC CP preprocessing requires eager calls, chunk_size=64 and ieee affine chains'
    if kwargs.get('MULTI_SEQS', False) or kwargs.get('USE_BG', False) or kwargs.get('bg') is not None:
        return False, 'AscendC CP preprocessing supports one GDN/KDA segment'
    if not backward and kwargs['v'] is not kwargs['u']:
        return False, 'AscendC CP preprocessing does not support separate DPLR values'
    shape = kwargs['k'].shape
    if len(shape) != 4 or shape[0] != 1 or shape[-1] != 128 or not 1 <= shape[2] <= 256:
        return False, 'AscendC CP preprocessing requires packed B=1, K=V=128 and at most 256 heads'
    if any(x.shape != shape or x.dtype != torch.bfloat16 for x in tensors):
        return False, 'AscendC CP preprocessing requires matching bfloat16 inputs and equal heads'
    if (g is None) == (gk is None):
        return False, 'AscendC CP preprocessing requires exactly one gate'
    gate = g if g is not None else gk
    gate_dtype = torch.bfloat16 if backward and gk is not None else torch.float32
    if gate.dtype != gate_dtype or gate.shape != (shape[:3] if g is not None else shape):
        return False, f'AscendC CP preprocessing requires matching {gate_dtype} gates'
    if output.shape != (shape[2], 128, 256) or output.dtype != torch.float32:
        return False, 'AscendC CP preprocessing requires a float32 affine summary'
    cu = kwargs['cu_seqlens']
    if cu is None or cu.ndim != 1 or cu.numel() != 2 or cu.dtype not in (torch.int32, torch.int64):
        return False, 'AscendC CP preprocessing requires a single segment boundary pair'
    return True, None


def _head_first(tensor):
    return None if tensor is None else tensor.transpose(1, 2).contiguous()


@register('cp', env_var='FLA_CP_ASCENDC')
class AscendCCPBackend(BaseBackend):
    backend_type = 'ascendc'
    package_name = 'fla_npu'
    default_enable = False
    priority = -1

    @classmethod
    def is_available(cls):
        return is_ascendc_available('FLA_CP_ASCENDC')

    def _pre_process_fwd_verifier(self, grid, use_graph=False, **kwargs):
        return _verify_pre_process(use_graph, kwargs, backward=False)

    def _pre_process_fwd(self, grid, use_graph=False, **kwargs):
        from fla_npu.ops.ascendc import pre_process_fwd_kernel_merged

        result = pre_process_fwd_kernel_merged(
            k=_head_first(kwargs['k']),
            w=_head_first(kwargs['w']),
            u=_head_first(kwargs['u']),
            g=_head_first(kwargs['g']),
            gk=_head_first(kwargs['gk']),
            cu_seqlens=kwargs['cu_seqlens'].tolist(),
            chunk_size=64,
        )
        kwargs['hm'].copy_(result[0])

    def _pre_process_bwd_verifier(self, grid, use_graph=False, **kwargs):
        return _verify_pre_process(use_graph, kwargs, backward=True)

    def _pre_process_bwd(self, grid, use_graph=False, **kwargs):
        from fla_npu.ops.ascendc import pre_process_bwd_kernel_merged

        result = pre_process_bwd_kernel_merged(
            q=_head_first(kwargs['q']),
            k=_head_first(kwargs['k']),
            w=_head_first(kwargs['w']),
            d_o=_head_first(kwargs['do']),
            dv=_head_first(kwargs['dv']),
            g=_head_first(kwargs['g']),
            gk=_head_first(kwargs['gk']),
            scale=kwargs['scale'],
            chunk_size=64,
            cu_seqlens=kwargs['cu_seqlens'].tolist(),
        )
        kwargs['dhm'].copy_(result)

    def _merge_states_verifier(self, grid, use_graph=False, **kwargs):
        h, summaries = kwargs['h'], kwargs['ag_hm']
        supported, reason = verify_ascendc_tensors(h, summaries)
        if not supported:
            return supported, reason
        if use_graph or kwargs['INTRACARD_MODE'] or kwargs['NUM_RANKS_ON_DEVICE']:
            return False, 'AscendC CP merge requires eager inter-rank calls'
        if kwargs.get('AFFINE_CHAIN_PRECISION') != 'ieee':
            return False, 'AscendC CP merge requires ieee affine chains'
        if any(kwargs.get(name) is not None for name in ('seq_offsets', 'init_offsets', 'h0_seq_ids', 'h0', 'h_seq_idx')):
            return False, 'AscendC CP merge does not support intracard sequence metadata'
        if h.ndim != 3 or h.shape[1:] != (128, 128) or not 1 <= h.shape[0] <= 256:
            return False, 'AscendC CP merge requires K=V=128 and at most 256 heads'
        if summaries.ndim != 4 or summaries.shape[1:] != (h.shape[0], 128, 256):
            return False, 'AscendC CP merge requires matching affine summaries'
        if not 1 <= summaries.shape[0] <= 1024:
            return False, 'AscendC CP merge supports at most 1024 rank summaries'
        if h.dtype != torch.float32 or summaries.dtype != torch.float32 or not h.is_contiguous():
            return False, 'AscendC CP merge requires float32 summaries and contiguous output'
        count, rank = kwargs['pre_or_post_num_ranks'], kwargs['rank']
        if not isinstance(count, int) or not isinstance(rank, int):
            return False, 'AscendC CP merge requires host rank indices'
        if not 1 <= count <= summaries.shape[0] or not 0 <= rank < summaries.shape[0]:
            return False, 'AscendC CP merge requires valid nonempty rank bounds'
        if (kwargs['FORWARD'] and count > rank) or (not kwargs['FORWARD'] and rank + count >= summaries.shape[0]):
            return False, 'AscendC CP merge rank window exceeds the available summaries'
        return True, None

    def _merge_states(self, grid, use_graph=False, **kwargs):
        from fla_npu.ops.ascendc import merge_fwd_bwd_kernel

        merge_fwd_bwd_kernel(
            h=kwargs['h'],
            ag_hm=kwargs['ag_hm'].contiguous(),
            pre_or_post_num_ranks=kwargs['pre_or_post_num_ranks'],
            rank=kwargs['rank'],
            forward=kwargs['FORWARD'],
            state_v_first=kwargs['STATE_V_FIRST'],
        )
