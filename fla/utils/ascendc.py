# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Shared availability and installation for optional AscendC wheel backends."""

import importlib
import logging
import os
import shlex
import subprocess
import sys
from functools import cache
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlsplit

from fla.utils import IS_NPU, find_spec_cached

logger = logging.getLogger(__name__)


def _get_wheel() -> tuple[str, str]:
    from packaging.requirements import Requirement
    from packaging.utils import parse_wheel_filename

    try:
        requirements = metadata.requires('flash-linear-attention') or []
    except metadata.PackageNotFoundError as error:
        raise ImportError('Install FLA first with python -m pip install -e . to expose its A5 wheel metadata.') from error
    for requirement in requirements:
        requirement = Requirement(requirement)
        if requirement.name != 'flash-linear-attention-npu-a5' or requirement.url is None:
            continue
        if requirement.marker is not None and not requirement.marker.evaluate({'extra': 'npu-ascendc-a5'}):
            continue
        url = urlsplit(requirement.url)
        if url.scheme != 'https' or not url.fragment.startswith('sha256=') or len(url.fragment) != 71:
            raise ImportError('The A5 wheel URL must use HTTPS and include a SHA-256 hash.')
        _, version, _, _ = parse_wheel_filename(unquote(url.path.rsplit('/', 1)[-1]))
        return requirement.url, str(version)
    raise ImportError('No A5 wheel matches this platform. Reinstall FLA metadata with python -m pip install -e .')


def _wheel_installed(expected_version: str) -> bool:
    for soc in ('a2', 'a3'):
        try:
            metadata.version(f'flash-linear-attention-npu-{soc}')
        except metadata.PackageNotFoundError:
            continue
        raise ImportError(f'An AscendC {soc.upper()} wheel is installed. Remove it before installing A5; they share fla_npu.')
    try:
        version = metadata.version('flash-linear-attention-npu-a5')
    except metadata.PackageNotFoundError:
        if find_spec_cached('fla_npu') is not None:
            raise ImportError('An unrecognized fla_npu installation exists. Remove it before installing the A5 wheel.')
        return False
    if version != expected_version:
        raise ImportError(f'AscendC A5 wheel {version} is installed, but FLA requires {expected_version}. Update it manually.')
    if find_spec_cached('fla_npu') is None:
        raise ImportError('The A5 wheel metadata exists but fla_npu is missing. Reinstall the A5 wheel.')
    return True


def ensure_ascendc_wheel() -> None:
    if not IS_NPU or os.environ.get('FLA_DISABLE_BACKEND_DISPATCH') == '1':
        return
    flags = (
        'FLA_ASCENDC', 'FLA_KDA_ASCENDC', 'FLA_KDA_ASCENDC_OPS', 'FLA_UTILS_ASCENDC',
        'FLA_COMMON_ASCENDC', 'FLA_GDN_ASCENDC', 'FLA_GLA_ASCENDC', 'FLA_CONV_ASCENDC', 'FLA_CP_ASCENDC',
    )
    if not any(os.environ.get(flag, '0') != '0' for flag in flags):
        return

    import torch
    from filelock import FileLock

    if 'Ascend950' not in torch.npu.get_device_name():
        logger.warning('Automatic AscendC installation supports A5 (Ascend950) only; trying other backends.')
        return

    wheel_url, version = _get_wheel()
    command = [sys.executable, '-m', 'pip', 'install', '--no-input', '--no-deps', wheel_url]
    environment_key = sha256(os.path.realpath(sys.prefix).encode()).hexdigest()
    lock_dir = Path.home() / '.cache' / 'fla'
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock_dir / f'ascendc-{environment_key}.lock'), timeout=660):
            # another importing process may have installed the wheel while we waited.
            importlib.invalidate_caches()
            find_spec_cached.cache_clear()
            if _wheel_installed(version):
                return
            logger.warning('Installing the AscendC A5 wheel into %s; this may take several minutes.', sys.prefix)
            subprocess.run(command, check=True, timeout=600)
            importlib.invalidate_caches()
            find_spec_cached.cache_clear()
            if not _wheel_installed(version):
                raise ImportError('pip completed without installing the expected A5 wheel.')
    except Exception as error:
        raise ImportError(
            f'AscendC A5 initialization failed: {error}. After resolving the cause, run: {shlex.join(command)}'
        ) from error


@cache
def _warn_missing_wheel() -> None:
    logger.warning(
        'AscendC was enabled but fla_npu is not installed; trying other backends. '
        'From the FLA source checkout, install exactly one wheel matching your NPU: '
        'A2 (910B): python -m pip install -e ".[npu-ascendc-a2]"; '
        'A3 (910_93): python -m pip install -e ".[npu-ascendc-a3]"; '
        'A5 (950): python -m pip install -e ".[npu-ascendc-a5]". '
        'Then restart Python. The adapters currently support A5 only.'
    )


def is_ascendc_available(env_var: str) -> bool:
    if not IS_NPU:
        return False
    if find_spec_cached('fla_npu') is not None:
        return True
    if os.environ.get('FLA_ASCENDC', '0') != '0' or os.environ.get(env_var, '0') != '0':
        _warn_missing_wheel()
    return False


def verify_ascendc_tensors(*tensors):
    import torch

    tensors = [tensor for tensor in tensors if tensor is not None]
    if torch.is_grad_enabled():
        return False, 'AscendC primitive requires disabled gradient recording'
    if not tensors or any(tensor.device.type != 'npu' for tensor in tensors):
        return False, 'AscendC requires NPU tensors'
    device = tensors[0].device
    if any(tensor.device != device or tensor.numel() == 0 for tensor in tensors):
        return False, 'AscendC requires nonempty tensors on one device'
    if 'Ascend950' not in torch.npu.get_device_name(device):
        return False, 'AscendC currently requires Ascend950'
    return True, None
