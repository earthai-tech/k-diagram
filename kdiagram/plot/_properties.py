from __future__ import annotations

from typing import Literal

import numpy as np
from matplotlib.axes import Axes

Acov = Literal[
    "default",
    "full",
    "full_circle",
    "half",
    "half_circle",
    "quarter",
    "quarter_circle",
    "eighth",
    "eighth_circle",
]

ZeroLoc = Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


TDG_DIRECTIONS = {
    "-1": {
        "CORR_POS": {
            "NE": [
                (1.0, 0.5),
                {"rotation_mode": "anchor", "rotation": 90},
            ],
            "SW": [
                (0.0, 0.5),
                {"rotation_mode": "anchor", "rotation": 90},
            ],
            "SE": [
                (0.5, 0.05),
                {"rotation": "horizontal"},
            ],
            "NW": [(0.5, 1.0), {"rotation": "horizontal"}],
            "W": [
                (0.0, 0.75),
                {"rotation_mode": "anchor", "rotation": 60},
            ],
            "E": [
                (1.0, 0.5),
                {"rotation_mode": "anchor", "rotation": 60},
            ],
            "N": [
                (1.0, 0.75),
                {"rotation_mode": "anchor", "rotation": -60},
            ],
            "S": [
                (0.0, 0.5),
                {"rotation_mode": "anchor", "rotation": -60},
            ],
        },
        "STD_POS": {
            "NE": [
                (0.25, 0.75),
                {"rotation_mode": "anchor", "rotation": 45},
            ],
            "SW": [
                (0.75, 0.25),
                {"rotation_mode": "anchor", "rotation": 45},
            ],
            "SE": [
                (0.75, 0.70),
                {"rotation_mode": "anchor", "rotation": -45},
            ],
            "NW": [
                (0.15, 0.35),
                {"rotation_mode": "anchor", "rotation": -45},
            ],
            "W": [
                (0.5, -0.1),
                {"rotation": "horizontal"},
            ],
            "E": [
                (0.5, 1.05),
                {"rotation": "horizontal"},
            ],
            "N": [
                (-0.1, 0.5),
                {"rotation": "vertical"},
            ],
            "S": [
                (1.1, 0.5),
                {"rotation": "vertical"},
            ],
        },
    },
    "1": {
        "CORR_POS": {
            "NW": [
                (0.0, 0.5),
                {"rotation_mode": "anchor", "rotation": 90},
            ],
            ###
            "NE": [
                (0.5, 0.95),
                {"rotation": "horizontal"},
            ],
            ##
            "SW": [
                (0.5, 0.05),
                {"rotation": "horizontal"},
            ],
            "SE": [(1.1, 0.5), {"rotation": "vertical"}],
            "W": [
                (0.20, 0.25),
                {"rotation_mode": "anchor", "rotation": -45},
            ],
            "E": [
                (0.8, 0.75),
                {"rotation_mode": "anchor", "rotation": -45},
                ##
            ],
            "N": [
                (0.1, 0.7),
                {"rotation_mode": "anchor", "rotation": 45},
            ],
            ##
            "S": [
                (0.85, 0.3),
                {"rotation_mode": "anchor", "rotation": 45},
            ],
        },
        "STD_POS": {
            #
            "NW": [
                (0.65, 0.85),
                {"rotation_mode": "anchor", "rotation": -45},
            ],
            "NE": [
                (0.90, 0.40),
                {"rotation_mode": "anchor", "rotation": 45},
            ],
            "SW": [
                (0.15, 0.65),
                {"rotation_mode": "anchor", "rotation": 45},
            ],
            "SE": [
                (0.25, 0.25),
                {"rotation_mode": "anchor", "rotation": -45},
            ],
            "W": [
                (0.5, 1.05),
                {"rotation": "horizontal"},
                ##
            ],
            "E": [
                (0.5, -0.1),
                {"rotation": "horizontal"},
            ],
            ##
            "N": [
                (1.1, 0.5),
                {"rotation": "vertical"},
            ],
            "S": [
                (-0.1, 0.5),
                {"rotation": "vertical"},
            ],
        },
    },
}


def _canon_acov(acov: Acov | str = "default") -> str:
    """Map aliases to canonical acov keys."""
    key = (acov or "default").lower().replace("-", "_")
    alias = {
        "default": "default",
        "full": "default",
        "full_circle": "default",
        "half": "half_circle",
        "half_circle": "half_circle",
        "quarter": "quarter_circle",
        "quarter_circle": "quarter_circle",
        "eighth": "eighth_circle",
        "eighth_circle": "eighth_circle",
    }
    return alias.get(key, key)


def _resolve_span(acov: str) -> float:
    """Return angular span (radians) for canonical acov."""
    spans = {
        "default": 2 * np.pi,
        "half_circle": 1 * np.pi,
        "quarter_circle": 0.5 * np.pi,
        "eighth_circle": 0.25 * np.pi,
    }
    if acov not in spans:
        raise ValueError(
            f"Invalid acov={acov!r}. Use one of "
            "{'default','half_circle','quarter_circle','eighth_circle'}."
        )
    return spans[acov]


# ---- base placement table (axes-fraction coords) ---------------------------
# For each acov we provide a sensible default for where the "theta/x" label
# (angular) and "r/y" label (radial) should go, depending on zero location.
# direction=+1 (CCW) is assumed; for -1 we mirror horizontally.
#
# Positions are (x, y) in Axes coordinates with text kwargs to avoid overlap.
# Feel free to tweak these empirically if you have a particular house style.
_POLAR_LABEL_TABLE: dict[
    str, dict[str, dict[str, tuple[tuple[float, float], dict]]]
] = {
    "quarter_circle": {
        # quarter is visually like a quadrant; these defaults keep labels clear
        "W": {
            "x": (
                (0.50, -0.10),
                dict(ha="center", va="top"),
            ),  # bottom center
            "y": (
                (-0.06, 0.50),
                dict(ha="right", va="center", rotation="vertical"),
            ),
        },
        "E": {
            "x": ((0.50, -0.10), dict(ha="center", va="top")),
            "y": (
                (1.06, 0.50),
                dict(ha="left", va="center", rotation="vertical"),
            ),
        },
        "N": {
            "x": ((0.50, 1.08), dict(ha="center", va="bottom")),
            "y": (
                (1.06, 0.50),
                dict(ha="left", va="center", rotation="vertical"),
            ),
        },
        "S": {
            "x": ((0.50, -0.10), dict(ha="center", va="top")),
            "y": (
                (-0.06, 0.50),
                dict(ha="right", va="center", rotation="vertical"),
            ),
        },
    },
    "half_circle": {
        # a semicircle — keep x along the long edge; y on the side
        "W": {
            "x": ((0.50, -0.10), dict(ha="center", va="top")),
            "y": (
                (-0.06, 0.50),
                dict(ha="right", va="center", rotation="vertical"),
            ),
        },
        "E": {
            "x": ((0.50, -0.10), dict(ha="center", va="top")),
            "y": (
                (1.06, 0.50),
                dict(ha="left", va="center", rotation="vertical"),
            ),
        },
        "N": {
            "x": ((0.50, 1.08), dict(ha="center", va="bottom")),
            "y": (
                (1.06, 0.50),
                dict(ha="left", va="center", rotation="vertical"),
            ),
        },
        "S": {
            "x": ((0.50, -0.12), dict(ha="center", va="top")),
            "y": (
                (-0.06, 0.50),
                dict(ha="right", va="center", rotation="vertical"),
            ),
        },
    },
    "default": {
        # full circle — angle label at bottom, radial on the right
        "W": {
            "x": ((0.50, -0.10), dict(ha="center", va="top")),
            "y": (
                (1.05, 0.50),
                dict(ha="left", va="center", rotation="vertical"),
            ),
        },
        "E": {
            "x": ((0.50, -0.10), dict(ha="center", va="top")),
            "y": (
                (-0.05, 0.50),
                dict(ha="right", va="center", rotation="vertical"),
            ),
        },
        "N": {
            "x": ((0.50, 1.08), dict(ha="center", va="bottom")),
            "y": (
                (1.05, 0.50),
                dict(ha="left", va="center", rotation="vertical"),
            ),
        },
        "S": {
            "x": ((0.50, -0.12), dict(ha="center", va="top")),
            "y": (
                (-0.05, 0.50),
                dict(ha="right", va="center", rotation="vertical"),
            ),
        },
    },
    "eighth_circle": {
        # very narrow span — keep labels close to the occupied edge
        "W": {
            "x": ((0.55, -0.10), dict(ha="center", va="top")),
            "y": (
                (-0.06, 0.45),
                dict(ha="right", va="center", rotation="vertical"),
            ),
        },
        "E": {
            "x": ((0.45, -0.10), dict(ha="center", va="top")),
            "y": (
                (1.06, 0.45),
                dict(ha="left", va="center", rotation="vertical"),
            ),
        },
        "N": {
            "x": ((0.50, 1.08), dict(ha="center", va="bottom")),
            "y": (
                (1.06, 0.45),
                dict(ha="left", va="center", rotation="vertical"),
            ),
        },
        "S": {
            "x": ((0.50, -0.12), dict(ha="center", va="top")),
            "y": (
                (-0.06, 0.45),
                dict(ha="right", va="center", rotation="vertical"),
            ),
        },
    },
}


def _mirror_if_clockwise(
    xy: tuple[float, float], direction: int
) -> tuple[float, float]:
    """If direction == -1 (clockwise), mirror horizontally in axes coords."""
    if direction == -1:
        return (1.0 - xy[0], xy[1])
    return xy


def place_polar_axis_labels(
    ax: Axes,
    *,
    x_label: str,
    y_label: str,
    acov: Acov | str = "quarter_circle",
    zero_location: ZeroLoc = "W",
    direction: int = 1,
    x_offset: tuple[float, float] = (0.0, 0.0),
    y_offset: tuple[float, float] = (0.0, 0.0),
    clear_default: bool = True,
    x_kw: dict | None = None,
    y_kw: dict | None = None,
):
    """
    Place angular (x) and radial (y) axis labels for a polar Axes with
    readable defaults across acov spans, zero locations, and directions.

    Returns
    -------
    x_text, y_text : matplotlib.text.Text
        The created text objects (useful for testing / tweaking).
    """
    canon = _canon_acov(acov)
    # fall back to 'default' if unusual zero_location is provided
    loc = zero_location if zero_location in _POLAR_LABEL_TABLE[canon] else "W"

    (x_xy, x_base_kw) = _POLAR_LABEL_TABLE[canon][loc]["x"]
    (y_xy, y_base_kw) = _POLAR_LABEL_TABLE[canon][loc]["y"]

    # Mirror horizontally if clockwise
    x_xy = _mirror_if_clockwise(x_xy, direction)
    y_xy = _mirror_if_clockwise(y_xy, direction)

    # Apply offsets (in axes fraction units)
    x_xy = (x_xy[0] + x_offset[0], x_xy[1] + x_offset[1])
    y_xy = (y_xy[0] + y_offset[0], y_xy[1] + y_offset[1])

    # Optionally clear default labels to avoid duplicates / overlaps
    if clear_default:
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Merge user kwargs
    x_draw_kw = {**x_base_kw, **(x_kw or {})}
    y_draw_kw = {**y_base_kw, **(y_kw or {})}

    x_text = ax.text(
        *x_xy,
        x_label,
        transform=ax.transAxes,
        clip_on=False,
        zorder=5,
        **x_draw_kw,
    )
    y_text = ax.text(
        *y_xy,
        y_label,
        transform=ax.transAxes,
        clip_on=False,
        zorder=5,
        **y_draw_kw,
    )
    return x_text, y_text
