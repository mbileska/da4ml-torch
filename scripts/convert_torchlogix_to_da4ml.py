#!/usr/bin/env python3
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path


class _Da4mlBuildArtifactFinder(importlib.abc.MetaPathFinder):
    def __init__(self, build_root: Path):
        suffixes = importlib.machinery.EXTENSION_SUFFIXES
        self._modules: dict[str, Path] = {}
        version_path = build_root / '_version.py'
        if version_path.exists():
            self._modules['da4ml._version'] = version_path
        for name in ('cmvm_bin', 'dais_bin'):
            for suffix in suffixes:
                artifact = build_root / f'{name}{suffix}'
                if artifact.exists():
                    self._modules[f'da4ml._binary.{name}'] = artifact
                    break

    def find_spec(self, fullname, path=None, target=None):
        artifact = self._modules.get(fullname)
        if artifact is None:
            return None
        return importlib.util.spec_from_file_location(fullname, artifact)


def _prefer_checkout_packages() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    da4ml_build = repo_root / 'da4ml' / 'build' / f'cp{sys.version_info.major}{sys.version_info.minor}'
    if da4ml_build.exists():
        marker = 'MESONPY_EDITABLE_SKIP'
        existing = [item for item in os.environ.get(marker, '').split(os.pathsep) if item]
        if str(da4ml_build) not in existing:
            os.environ[marker] = os.pathsep.join([*existing, str(da4ml_build)])
        sys.meta_path.insert(0, _Da4mlBuildArtifactFinder(da4ml_build))

    for path in (
        repo_root,
        repo_root / 'src',
        repo_root / 'torchlogix' / 'src',
        repo_root / 'da4ml' / 'src',
    ):
        if path.exists():
            sys.path.insert(0, str(path))


_prefer_checkout_packages()

from da4ml_torch.tools.torchlogix_to_da4ml import main  # noqa: E402


if __name__ == '__main__':
    raise SystemExit(main())
