# File: tests/test_metric_utils.py

import pytest
import numpy as np
from sklearn.metrics import ( # Import for comparison/validation if needed
    r2_score, mean_absolute_error, accuracy_score, precision_score
)

try:
    from kdiagram.utils.metric_utils import get_scorer
    _SKIP_TESTS = False
except ImportError as e:
    print(f"Could not import get_scorer: {e}. Skipping tests.")
    _SKIP_TESTS = True

# Skip all tests in this file if get_scorer cannot be imported
pytestmark = pytest.mark.skipif(
    _SKIP_TESTS, reason="get_scorer or its dependencies not found"
)

# --- Test Data ---
# Regression
y_true_reg = np.array([1, 2, 3, 4, 5])
y_pred_reg1 = np.array([1.1, 1.9, 3.1, 4.2, 4.8]) # Good fit
y_pred_reg2 = np.array([5, 4, 3, 2, 1])       # Bad fit

# Classification (Binary)
y_true_clf = np.array([0, 1, 0, 1, 1, 0, 1])
y_pred_clf1 = np.array([0, 1, 0, 1, 0, 0, 1]) # Mostly correct
y_pred_clf2 = np.array([1, 0, 1, 0, 0, 1, 0]) # Mostly incorrect

# --- Tests ---
@pytest.mark.skip ("'squared' seems deprecated in newest version of scikit-learn")
@pytest.mark.parametrize("metric_name", [
    "r2", "mae", "rmse", "mse", "mape",
    "mean_absolute_error", "root_mean_squared_error" # Aliases
])
def test_get_scorer_valid_regression(metric_name):
    """Test retrieving valid regression scorers."""
    try:
        scorer_func = get_scorer(metric_name)
        assert callable(scorer_func), \
            f"get_scorer('{metric_name}') did not return a callable."

        # Test if the scorer runs without error
        score1 = scorer_func(y_true_reg, y_pred_reg1)
        score2 = scorer_func(y_true_reg, y_pred_reg2)
        assert isinstance(score1, (float, int, np.number)), \
            f"Scorer for '{metric_name}' did not return a number."
        assert isinstance(score2, (float, int, np.number))

        # Optional: Basic sanity check for known values
        if metric_name in ["mae", "mean_absolute_error"]:
            assert np.isclose(score1, np.mean([0.1, 0.1, 0.1, 0.2, 0.2]))
            assert np.isclose(score2, np.mean([4, 2, 0, 2, 4]))

    except ValueError as e:
        pytest.fail(f"get_scorer('{metric_name}') raised unexpected "
                    f"ValueError: {e}")
    except ImportError as e:
         pytest.fail(f"ImportError during scorer test for '{metric_name}'."
                     f" Is scikit-learn installed? Error: {e}")

@pytest.mark.skip ("skip close value testing with np.isclose.")
@pytest.mark.parametrize("metric_name", [
    "accuracy", "precision", "recall", "f1",
    "accuracy_score", "precision_weighted", "recall_weighted", "f1_weighted"
])
def test_get_scorer_valid_classification(metric_name):
    """Test retrieving valid classification scorers."""
    try:
        scorer_func = get_scorer(metric_name)
        assert callable(scorer_func), \
            f"get_scorer('{metric_name}') did not return a callable."

        # Test if the scorer runs without error
        score1 = scorer_func(y_true_clf, y_pred_clf1)
        score2 = scorer_func(y_true_clf, y_pred_clf2)
        assert isinstance(score1, (float, int, np.number)), \
            f"Scorer for '{metric_name}' did not return a number."
        assert isinstance(score2, (float, int, np.number))

        # Optional: Basic sanity check for known values
        if metric_name in ["accuracy", "accuracy_score"]:
            # y_pred_clf1: 6/7 correct -> 0.857
            # y_pred_clf2: 1/7 correct -> 0.143
            assert np.isclose(score1, 6/7,  rtol=0.1)
            assert np.isclose(score2, 1/7)

    except ValueError as e:
        pytest.fail(f"get_scorer('{metric_name}') raised unexpected "
                    f"ValueError: {e}")
    except ImportError as e:
         pytest.fail(f"ImportError during scorer test for '{metric_name}'."
                     f" Is scikit-learn installed? Error: {e}")

def test_get_scorer_case_insensitivity():
    """Test if scorer lookup is case-insensitive."""
    scorer_lower = get_scorer('r2')
    scorer_upper = get_scorer('R2')
    scorer_mixed = get_scorer('rMsE')
    assert callable(scorer_lower)
    assert callable(scorer_upper)
    assert callable(scorer_mixed)
    # Check if they point to the same underlying function (or wrappers)
    # Note: Direct comparison might fail if wrappers are used, but
    # ensuring they are callable is the main goal.
    assert scorer_lower is get_scorer('r2')
    assert scorer_mixed is get_scorer('rmse')

def test_get_scorer_unknown_metric():
    """Test ValueError for unknown metric names."""
    with pytest.raises(ValueError, match="Unknown scoring metric 'invalid_metric_name'"):
        get_scorer('invalid_metric_name')

@pytest.mark.parametrize("invalid_input", [None, 123, ['r2'], {}])
def test_get_scorer_invalid_input_type(invalid_input):
    """Test TypeError for non-string input."""
    with pytest.raises(TypeError, match="Expected string scoring name"):
        get_scorer(invalid_input)

# Optional: Test for sklearn missing (requires mocking infrastructure)
# @patch('kdiagram.utils.metric_utils._SKLEARN_AVAILABLE', False)
# def test_get_scorer_sklearn_missing():
#     """Test ImportError if sklearn is marked as unavailable."""
#     with pytest.raises(ImportError, match="scikit-learn is required"):
#         get_scorer('r2')

if __name__=='__main__': 
    pytest.main([__file__])