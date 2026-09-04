"""The in-process Python linter must catch undefined names, not just syntax.

Session 20260904_000725_c7a0fe wrote a ``vault_web.py`` using ``os.environ``
without ``import os``; ``ast.parse`` accepted it, pyright never ran because
the file sat outside any git worktree, and the bug was only found by a later
manual ``grep`` after the service had been started. The fix is a fast ruff
F821/F822/F823 pass after ``ast.parse`` — no project root required.
"""

import pytest

from tools.file_operations import _find_ruff, _lint_python_inproc

needs_ruff = pytest.mark.skipif(_find_ruff() is None, reason="ruff not installed")


def test_syntax_error_still_reported_first():
    ok, err = _lint_python_inproc("def f(:\n    pass\n")
    assert not ok
    assert err.startswith("SyntaxError")


def test_clean_file_passes():
    ok, err = _lint_python_inproc("import os\n\ndef f():\n    return os.getcwd()\n")
    assert ok, err


@needs_ruff
def test_undefined_name_is_reported():
    ok, err = _lint_python_inproc("def f():\n    return os.getcwd()\n")
    assert not ok
    assert ("F821" in err or "undefined-name" in err) and "`os`" in err


@needs_ruff
def test_style_rules_are_not_reported():
    # Unused import (F401) and long line (E501) must NOT fail the write.
    ok, err = _lint_python_inproc("import json\nx = 1  # " + "y" * 200 + "\n")
    assert ok, err


@needs_ruff
def test_builtins_and_dunder_are_not_flagged():
    ok, err = _lint_python_inproc(
        "print(__file__, __name__, len([]))\n"
        "try:\n    pass\nexcept Exception as e:\n    print(e)\n"
    )
    assert ok, err
