#!/usr/bin/env python3
from __future__ import annotations

from convert_torchlogix_to_da4ml import _prefer_checkout_packages


_prefer_checkout_packages()

from da4ml_torch.tools.torchlogix_to_da4ml import estimate_main  # noqa: E402


if __name__ == '__main__':
    raise SystemExit(estimate_main())
