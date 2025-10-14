'''
##############################################################
# Created Date: Wednesday, October 1st 2025
# Contact Info: luoxiangyong01@gmail.com
# Author/Copyright: Mr. Xiangyong Luo
##############################################################
'''


import sys
import pytest
from realtwin import check_python_version
from io import StringIO


def make_sys_version(version_str):
    """Helper to patch sys.version to a specific version string."""
    return version_str + " (default, Jan 1 2024, 00:00:00) [MSC v.1916 64 bit (AMD64)]"


@pytest.mark.parametrize(
    "sys_version,min_version,expected_tuple,should_raise",
    [
        ("3.10.0", "3.10", (3, 10, 0), False),
        ("3.11.2", "3.10", (3, 11, 2), False),
        ("3.9.9", "3.10", (3, 9, 9), True),
        ("3.10.1", "3.10", (3, 10, 1), False),
        ("3.10.0", "3.11", (3, 10, 0), True),
        ("3.11.0", "3.11", (3, 11, 0), False),
        ("3.12.0", "3.11", (3, 12, 0), False),
    ]
)
def test_check_python_version(monkeypatch, sys_version, min_version, expected_tuple, should_raise):
    # Patch sys.version
    monkeypatch.setattr(sys, "version", make_sys_version(sys_version))
    if should_raise:
        # The function prints a message and returns the tuple, does not raise
        # So we capture stdout to check the print
        captured = StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        result = check_python_version(min_version)
        assert result == expected_tuple
        output = captured.getvalue()
        assert f"pyufunc supports Python {min_version} or higher." in output
    else:
        result = check_python_version(min_version)
        assert result == expected_tuple


def test_check_python_version_default(monkeypatch):
    # Should use current sys.version and min_version="3.10"
    result = check_python_version()
    # Should return a tuple of ints
    assert isinstance(result, tuple)
    assert all(isinstance(x, int) for x in result)
