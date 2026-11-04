"""This package must not claim another project's PyPI name.

`tiny-agent-memory` (0.1.0) and `tiny-agent-memory-v2` (2.0.0) are two
different projects: different layouts, different code, different READMEs.
But both declared `name = "tiny-agent-memory"` in pyproject.toml, so only
one of them could ever be published, and `pip install tiny-agent-memory`
was ambiguous.

This repo is the v2 successor, so it takes the name that matches its own
repository and README: `tiny-agent-memory-v2`.

The guard exists because the name is stated in three places -- the
pyproject, and two `pip install` lines in the README -- and the README
lines are easy to miss.
"""

from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PACKAGE_NAME = "tiny-agent-memory-v2"
OTHER_PROJECT = "tiny-agent-memory"


def _read(name: str) -> str:
    with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
        return fh.read()


def test_pyproject_declares_the_v2_package_name():
    text = _read("pyproject.toml")
    match = re.search(r'^\s*name\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "no name= found in pyproject.toml"
    assert match.group(1) == PACKAGE_NAME, (
        "pyproject.toml declares %r; this repository is %r"
        % (match.group(1), PACKAGE_NAME)
    )


def test_readme_installs_the_v2_package_name():
    """Every `pip install` in the README must name this package."""
    text = _read("README.md")
    installs = re.findall(r"pip install ([A-Za-z0-9._\-]+)", text)
    assert installs, "no pip install line found in README.md"
    for name in installs:
        assert name == PACKAGE_NAME, (
            "README says `pip install %s` but this package is %s. "
            "The other project owns %s."
            % (name, PACKAGE_NAME, OTHER_PROJECT)
        )


def test_readme_title_matches_the_package_name():
    text = _read("README.md")
    heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    assert heading, "no H1 in README.md"
    assert heading.group(1).strip().startswith(PACKAGE_NAME), (
        "README H1 is %r but the package is %r"
        % (heading.group(1).strip(), PACKAGE_NAME)
    )
