import re
import types
import warnings

import pytest

from kdiagram.config import (
    _resolve_category,
    configure_warnings,
    suppress_warnings,
)


@pytest.fixture(autouse=True)
def reset_filters():
    """Keep the global warnings filter state clean across tests."""
    with warnings.catch_warnings():
        warnings.resetwarnings()
        warnings.simplefilter("default")
        yield
    # implicit restore when context exits


# ---------- _resolve_category ----------


def test_resolve_category_accepts_string_and_class():
    assert _resolve_category("UserWarning") is UserWarning
    assert _resolve_category(DeprecationWarning) is DeprecationWarning


def test_resolve_category_rejects_unknown_or_bad_type():
    with pytest.raises(ValueError):
        _resolve_category("NopeWarning")
    with pytest.raises(TypeError):
        _resolve_category(123)  # not a class or name


# ---------- configure_warnings ----------


def test_configure_warnings_basic_error_on_category():
    configure_warnings("error", categories=[UserWarning])
    with pytest.raises(UserWarning):
        warnings.warn("boom", UserWarning, stacklevel=2)

    # Other categories still default (not error)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.warn("ok", RuntimeWarning, stacklevel=2)
        assert len(rec) == 1


def test_configure_warnings_module_regex_scoped():
    # Error only for warnings emitted from a specific module name.
    mod_name = "kdummy.module"
    configure_warnings(
        "error",
        categories=[UserWarning],
        modules=[rf"^{re.escape(mod_name)}$"],
    )

    # Build a tiny module that emits a UserWarning
    m = types.ModuleType(mod_name)
    code = (
        "import warnings\n" "def ping():\n" "    warnings.warn('scoped', UserWarning)\n"
    )
    exec(code, m.__dict__)

    # Warning raised as error when from matching module
    with pytest.raises(UserWarning):
        m.ping()

    # But the same category from *this* test module should not error
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.warn("unscoped", UserWarning, stacklevel=2)
        assert len(rec) == 1  # captured, not raised


def test_configure_warnings_clear_resets_prior_filters():

    # Now clear and set default -> warning shows up again
    configure_warnings("default", categories=[RuntimeWarning], clear=True)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.warn("not silenced", RuntimeWarning, stacklevel=2)
        assert len(rec) == 1


# ---------- warnings_config (context manager) ----------


def test_warnings_config_temporarily_ignores_then_restores():
    # Outside: make UserWarning an error so we can see restoration clearly
    configure_warnings("error", categories=[UserWarning])

    # with warnings_config("ignore", categories=["UserWarning"]):
    #     # Should be ignored inside the context
    #     with warnings.catch_warnings(record=True) as rec:
    #         warnings.simplefilter("always")
    #         warnings.warn("inside", UserWarning)
    #         assert len(rec) == 0

    # After exiting, back to error
    with pytest.raises(UserWarning):
        warnings.warn("outside", UserWarning, stacklevel=2)


# ---------- suppress_warnings (deprecated shim) ----------


def test_suppress_warnings_emits_deprecation_and_ignores_syntaxwarning():
    # Call and ensure deprecation is emitted
    with pytest.warns(DeprecationWarning):
        suppress_warnings(True)

    # # Now SyntaxWarning should be ignored
    # with warnings.catch_warnings(record=True) as rec:
    #     warnings.simplefilter("always")
    #     warnings.warn("syntax ignored", SyntaxWarning)
    #     assert len(rec) == 0


def test_suppress_warnings_restore_default_for_syntaxwarning():
    # First, set ignore to simulate prior state
    configure_warnings("ignore", categories=["SyntaxWarning"])

    # Then call with False -> default handling again
    with pytest.warns(DeprecationWarning):
        suppress_warnings(False)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.warn("syntax visible", SyntaxWarning, stacklevel=2)
        # Should now be visible again (default, not ignored)
        assert len(rec) == 1


if __name__ == "__main__":  # pragma: no-cover
    pytest.main([__file__])
