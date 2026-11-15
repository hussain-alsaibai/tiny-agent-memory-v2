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


GIT_URL = "^git\\+https://github\\.com/(?:[^/]+)/([^/@\\s\"']+?)(?:\\.git)?$"

# Two install forms are correct here:
#   pip install tiny-agent-memory-v2                       (once published)
#   pip install git+https://github.com/<owner>/<repo>.git   (today)
# Anything else -- a bare name that belongs to another project, or a git URL
# pointing at a different repository -- is a bug.
INSTALL_RE = re.compile(
    r'^\s*pip install (?:\\\n\s*)*(?:"([^"]+)"|\'([^\']+)\'|(\S+))',
    re.MULTILINE,
)


def _installed_targets(text):
    for m in INSTALL_RE.finditer(text):
        target = next(g for g in m.groups() if g is not None)
        # strip PEP 508 extras / direct-URL parts down to the thing being installed
        if target.startswith("git+") or " @ " in target:
            yield ("git", target)
        else:
            yield ("name", target.split("==")[0].split("[")[0].strip())


def test_readme_installs_the_v2_package_name():
    """Every `pip install` in the README names this package, by name or URL."""
    text = _read("README.md")
    targets = list(_installed_targets(text))
    assert targets, "no pip install line found in README.md"
    for kind, target in targets:
        if kind == "name":
            assert target == PACKAGE_NAME, (
                "README says `pip install %s` but this package is %s. "
                "The other project owns %s."
                % (target, PACKAGE_NAME, OTHER_PROJECT)
            )
        else:
            url = target.split(" @ ")[-1]
            m = re.match(GIT_URL, url)
            assert m and m.group(1) == PACKAGE_NAME, (
                "README installs from %r, which is not this repository (%s)"
                % (url, PACKAGE_NAME)
            )


def test_readme_title_matches_the_package_name():
    text = _read("README.md")
    heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    assert heading, "no H1 in README.md"
    assert heading.group(1).strip().startswith(PACKAGE_NAME), (
        "README H1 is %r but the package is %r"
        % (heading.group(1).strip(), PACKAGE_NAME)
    )
