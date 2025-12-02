from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pytest

import sys


def run_all(args: Iterable[str] | None = None) -> int:
	"""Execute every test module under ``src/testing`` using ``pytest``.

	Parameters
	----------
	args:
		Optional extra CLI-style arguments to pass through to ``pytest``.

	Returns
	-------
	int
		The exit code reported by ``pytest.main`` (0 indicates success).
	"""

	test_dir = Path(__file__).parent
	base_args: Sequence[str]
	if args is None:
		base_args = [str(test_dir)]
	else:
		base_args = [str(arg) for arg in args]
		if str(test_dir) not in base_args:
			base_args = list(base_args) + [str(test_dir)]
	return pytest.main(list(base_args))


__all__ = ["run_all"]


if __name__ == "__main__":
    sys.exit(run_all())

