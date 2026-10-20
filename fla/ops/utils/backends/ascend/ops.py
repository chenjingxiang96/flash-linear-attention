# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.utils.ascendc import verify_ascendc_tensors


def chunk_local_cumsum_verifier(
    g, chunk_size, reverse=False, scale=None, cu_seqlens=None, output_dtype=torch.float,
    chunk_indices=None, use_graph=False, **kwargs,
):
    supported, reason = verify_ascendc_tensors(g)
    if not supported:
        return supported, reason
    if use_graph or cu_seqlens is not None or chunk_indices is not None or kwargs:
        return False, 'AscendC cumsum currently supports dense eager calls'
    if g.ndim not in (3, 4) or chunk_size not in (16, 32, 64, 128):
        return False, 'AscendC cumsum requires rank 3/4 and chunk_size=16/32/64/128'
    if g.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False, 'AscendC cumsum requires float16/bfloat16/float32'
    if (output_dtype or g.dtype) not in (torch.float16, torch.bfloat16, torch.float32):
        return False, 'AscendC cumsum output must be float16/bfloat16/float32'
    return True, None


def chunk_local_cumsum(
    g, chunk_size, reverse=False, scale=None, cu_seqlens=None, output_dtype=torch.float,
    chunk_indices=None, use_graph=False, **kwargs,
):
    from fla_npu.ops.ascendc import chunk_local_cumsum as native_cumsum

    result = native_cumsum(
        g=g.reshape(g.shape[0], g.shape[1], -1).transpose(1, 2).contiguous(),
        chunk_size=chunk_size,
        reverse=reverse,
        scale=1.0 if scale is None else scale,
        head_first=True,
        output_dtype='float32',
    )
    return result.transpose(1, 2).reshape(g.shape).to(output_dtype or g.dtype).contiguous()


def solve_tril_verifier(A, cu_seqlens=None, chunk_indices=None, output_dtype=torch.float):
    supported, reason = verify_ascendc_tensors(A)
    if not supported:
        return supported, reason
    if cu_seqlens is not None or chunk_indices is not None:
        return False, 'AscendC solve currently supports dense inputs'
    if A.ndim != 4 or A.shape[-1] not in (16, 32, 64):
        return False, 'AscendC solve requires [B, T, H, BT] with BT=16/32/64'
    if A.dtype not in (torch.float16, torch.bfloat16) or (output_dtype or A.dtype) != A.dtype:
        return False, 'AscendC solve requires matching float16/bfloat16 input and output'
    return True, None


def solve_tril(A, cu_seqlens=None, chunk_indices=None, output_dtype=torch.float):
    from fla_npu.ops.ascendc import solve_tri

    return solve_tri(x=A.contiguous(), layout='bsnd')
