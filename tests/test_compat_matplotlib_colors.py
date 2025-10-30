import pytest

from kdiagram.compat.matplotlib import get_cmap, get_colors, is_valid_cmap


def test_get_cmap_valid():
    # Test for valid colormap names
    cmap = get_cmap("viridis")
    assert cmap is not None, "Colormap 'viridis' should be valid"

    cmap = get_cmap("tab10")
    assert cmap is not None, "Colormap 'tab10' should be valid"


def test_get_cmap_invalid():
    # Test for invalid colormap name
    with pytest.warns(UserWarning):
        cmap = get_cmap("invalid_cmap")
    assert cmap is not None, (
        "Invalid colormap name should fall back to default"
    )

    # Test for None input with allow_none=False (default behavior)
    with pytest.warns(UserWarning):
        cmap = get_cmap(None)
    assert cmap is not None, "None should fallback to a default colormap"


def test_get_colors_single_color():
    # Test for a single color input
    colors = get_colors(3, "blue")
    assert len(colors) == 3, "Should return 3 colors"
    assert colors[0] == "blue", "The first color should be 'blue'"


def test_get_colors_multiple_colors():
    # Test for multiple colors
    colors = get_colors(3, ["blue", "orange"])
    assert len(colors) == 3, "Should return 3 colors"
    assert colors[0] == "blue", "The first color should be 'blue'"
    assert colors[1] == "orange", "The second color should be 'orange'"


def test_get_colors_invalid():
    # Test for an invalid color input
    colors = get_colors(3, "invalid_color")
    assert len(colors) == 3, "Should return 3 colors even for invalid color"
    assert colors[0] == "invalid_color", (
        "The first color should be the user-provided color"
    )


def test_is_valid_cmap_valid():
    # Test for valid colormap name
    assert is_valid_cmap("viridis") is True, "Should be valid colormap"
    assert is_valid_cmap("tab10") is True, "Should be valid colormap"


def test_is_valid_cmap_invalid():
    # Test for invalid colormap name
    assert is_valid_cmap("invalid_cmap") is False, (
        "Should be invalid colormap"
    )
    assert is_valid_cmap("notacmap") is False, "Should be invalid colormap"


@pytest.mark.parametrize(
    "n, colors, expected_length",
    [
        (3, "blue", 3),
        (5, ["blue", "green", "red"], 5),
        (2, None, 2),
    ],
)
def test_get_colors_parametrized(n, colors, expected_length):
    colors_list = get_colors(n, colors)
    assert len(colors_list) == expected_length, (
        f"Should return {expected_length} colors"
    )
