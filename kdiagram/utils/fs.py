from __future__ import annotations

import os
from pathlib import Path
import warnings
from typing import Any, Literal
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _unique_path(outdir: Path, stem: str, ext: str) -> Path:
    """Return a non-colliding path like 'stem.ext', 'stem (1).ext', ..."""
    cand = outdir / f"{stem}{ext}"
    i = 1
    while cand.exists():
        cand = outdir / f"{stem} ({i}){ext}"
        i += 1
    return cand


def savefig(
    savefig: str | os.PathLike | None,
    fig_or_ax: Figure | Axes | None = None,   
    *,
    dpi: int = 300,
    bbox_inches: str = "tight",
    pad_inches: float = 0.2,
    facecolor: Any | None = None,
    edgecolor: Any | None = None,
    overwrite: bool = False,
    error: str = "warn",  
    close: Literal["auto", True, False] = "auto",
    **kwargs,
) -> Path | None:
    
    if savefig is None:
        return None

    # --- Resolve the figure to save (exactly once) ---
    # We also track whether we auto-fetched a figure so that the caller can
    # use 'close="auto"' semantics.
    auto_fetched = False
    
    if isinstance(fig_or_ax, Axes):
        fig = fig_or_ax.figure
    elif isinstance(fig_or_ax, Figure):
        fig = fig_or_ax
    else:
        # fig_or_ax is None (or not a Figure/Axes): try to grab an existing fig
        # without creating a new empty one.
        fignums = plt.get_fignums()
        if not fignums:
            warnings.warn(
                "No active Matplotlib figure to save. Pass a Figure or Axes "
                "explicitly (fig_or_ax=...).",
                stacklevel=2,
            )
            return None
        fig = plt.figure(fignums[-1])   # attach to the last active figure
        auto_fetched = True

    # --- Normalize path and choose output file name ---
    raw = str(savefig)
    path = Path(os.path.expanduser(os.path.expandvars(raw)))

    is_dir_hint = raw.endswith(("/", "\\"))  # user typed a folder path
    if (path.exists() and path.is_dir()) or is_dir_hint:
        outdir = path
        stem, ext = "figure", ".png"
    else:
        outdir = path.parent
        stem = path.stem or "figure"
        ext = path.suffix or ".png"

    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        msg = f"Could not create directory '{outdir}': {e}"
        if error == "raise":
            raise
        if error == "warn":
            warnings.warn(msg, stacklevel=2)
        return None

    def _unique_path(d: Path, s: str, e: str) -> Path:
        p = d / f"{s}{e}"
        if overwrite or not p.exists():
            return p
        i = 1
        while True:
            cand = d / f"{s} ({i}){e}"
            if not cand.exists():
                return cand
            i += 1

    final = (
        outdir / f"{stem}{ext}") if overwrite else _unique_path(
            outdir, stem, ext)

    # --- Save ---
    try:
        # Don't fight constrained_layout
        if not fig.get_constrained_layout():
            fig.tight_layout()

        fig.savefig(
            final,
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            facecolor=facecolor,
            edgecolor=edgecolor,
            **kwargs,
        )
        print(f"===> Plot saved to {final}")
        
        # Smart close
        should_close = (
            True if close is True
            else False if close is False
            else auto_fetched             # close only if we grabbed gcf()
        )
        if should_close:
            plt.close(fig)
            
        return final
    

    except Exception as e:
        msg = f"Failed to save figure to '{final}': {e}"
        if error == "raise":
            raise
        if error == "warn":
            warnings.warn(msg, stacklevel=2)
        return None

savefig.__doc__=r"""
Save a Matplotlib figure robustly.

This helper wraps ``Figure.savefig`` with safe path handling,
directory creation, unique file naming, and optional smart figure
closing. It can also save the *current* active figure when no
``Figure`` or ``Axes`` is provided.

Parameters
----------
savefig : str or Path or None
    Target path. If ``None``, nothing is saved and ``None`` is
    returned. If a directory is given (or the string ends with
    ``'/'`` or ``'\\'``), a default filename ``'figure.png'`` is
    used inside that directory. If no extension is present, ``.png``
    is assumed.
fig_or_ax : Figure or Axes or None, optional
    A Matplotlib ``Figure`` or ``Axes`` to save. If an ``Axes`` is
    passed, its parent figure is used. If ``None``, the helper tries
    to resolve the most recently active figure (see Notes). If no
    figure is active, a warning is issued and ``None`` is returned.
dpi : int, default=300
    Resolution passed to ``Figure.savefig``.
bbox_inches : str, default='tight'
    Bounding box option forwarded to ``Figure.savefig``.
pad_inches : float, default=0.2
    Padding when ``bbox_inches='tight'``.
facecolor, edgecolor : Any, optional
    Colors forwarded to ``Figure.savefig``.
overwrite : bool, default=False
    If ``True``, overwrite an existing file. If ``False``, create a
    unique name by appending ``' (1)'``, ``' (2)'``, ... before the
    extension.
error : {'warn', 'raise', 'ignore'}, default='warn'
    How to handle I/O errors (directory creation or disk write).
    ``'warn'`` emits a ``UserWarning`` and returns ``None`` on
    failure; ``'raise'`` re-raises; ``'ignore'`` quietly returns
    ``None``.
close : {'auto', True, False}, default='auto'
    Figure closing policy after a successful save.

    * ``'auto'``: close only if the figure was auto-fetched (i.e.,
      ``fig_or_ax`` was ``None`` and an existing active figure was
      used). This is convenient when calling ``kd.savefig(...)`` at
      top level.
    * ``True``: always close the figure that was saved, regardless
      of how it was obtained (explicit or auto-fetched).
    * ``False``: never close the figure; the caller manages figure
      lifecycle.

**kwargs
    Additional keyword arguments are forwarded to
    ``Figure.savefig``.

Returns
-------
Path or None
    The final saved path as a ``pathlib.Path`` when successful;
    otherwise ``None`` (e.g., ``savefig is None`` or no active
    figure was found).

Notes
-----
* If ``fig_or_ax`` is ``None``, the helper resolves the last active
  figure using ``matplotlib.pyplot.get_fignums`` and
  ``matplotlib.pyplot.figure``. If no figures exist, a warning is
  emitted and ``None`` is returned.
* When ``overwrite`` is ``False`` and the target exists, the helper
  creates a unique filename by appending a space and a counter in
  parentheses before the extension.
* If the figure uses constrained layout, the helper will not call
  ``tight_layout``; otherwise it applies ``tight_layout`` before
  saving to reduce label overlap.
* To avoid external labels being cropped when using
  ``bbox_inches='tight'``, prefer placing labels inside axes, or use
  figure-level labels (e.g., ``Figure.suptitle``, ``Figure.supylabel``).
* This function does not create an empty figure; it only saves an
  existing one.

Examples
--------
Save the current active figure and auto-close it::

    import matplotlib.pyplot as plt
    import kdiagram as kd
    plt.plot([0, 1], [0, 1])
    kd.savefig("out/line.png")  # auto-fetched, then closed

Save an explicit figure and keep it open::

    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    kd.savefig("results/plot.png", fig, close=False)

Ensure overwrite and force close::

    kd.savefig("results/plot.png", fig, overwrite=True, close=True)

Save from an Axes object (parent figure is used)::

    kd.savefig("results/axes_plot.png", ax)

Let the helper pick a default filename in a directory path::

    kd.savefig("results/")  # creates results/figure.png (or figure (1).png)

See Also
--------
matplotlib.figure.Figure.savefig
matplotlib.pyplot.gcf
matplotlib.pyplot.get_fignums

References
----------
.. [1] Matplotlib Figure.savefig documentation.
.. [2] Matplotlib tight_layout and constrained_layout guides.
.. [3] Python pathlib documentation for path handling.
"""
