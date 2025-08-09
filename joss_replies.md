Thanks so much for the thoughtful, detailed review and for taking the time to go through the repo 
so carefully. Your notes were spot-on and very actionable. Below I respond point-by-point 
for the first two sections, including exactly what I changed.

# Documentation

***1) A11y (acessibility): When hovering over an item in the left table of contents, the contrast between font (turning white) and background is not high enough. How about changing the font to black when hovering?***

**Action taken** — I reworked the sidebar hover/active styles to remove the Furo gradient and use a solid background with high-contrast text. Hover and active states now set a solid `background-color: var(--color-brand-primary)` and force the link color to `var(--color-background-primary)` (white on dark teal). This yields AA+ contrast and is much clearer than the previous gradient. Relevant CSS:

```css
.sidebar-drawer .reference:hover {
  background-image: none !important;
  background-color: var(--color-brand-primary) !important;
  color: var(--color-background-primary) !important;
  padding-left: 1.1em;
}
.sidebar-drawer li.current > a.reference {
  background-image: none !important;
  background-color: var(--color-brand-primary) !important;
  color: var(--color-background-primary) !important;
  font-weight: 600;
}
```

***2) It seems that docstrings use numpydoc, but the Sphinx documentation does not use the extension ([https://numpydoc.readthedocs.io/en/latest/index.html](https://numpydoc.readthedocs.io/en/latest/index.html)), even though napoleon\_numpy\_docstring = True is set, resulting in Reference pages where the docstrings are not parsed correctly (see e.g. [https://k-diagram.readthedocs.io/en/latest/\_autosummary/uncertainty/kdiagram.plot.uncertainty.plot\_anomaly\_magnitude.html](https://k-diagram.readthedocs.io/en/latest/_autosummary/uncertainty/kdiagram.plot.uncertainty.plot_anomaly_magnitude.html))***

**Action taken** — I installed and enabled the **numpydoc** extension in `conf.py` (alongside Napoleon), then fixed a few docstring issues (duplicate “Notes” sections and indentation) that were causing parse errors. Autosummary/API pages like `plot_anomaly_magnitude` now render Parameters/Notes/Examples and math blocks correctly.

```diff
extensions = [
+   'numpydoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    ...
]
```

# Packaging

***1) Package uses setup.py which is not deprecated, but using a pyproject.toml is preferable. See [https://packaging.python.org/en/latest/guides/modernize-setup-py-project/](https://packaging.python.org/en/latest/guides/modernize-setup-py-project/)***

**Action taken** — I migrated to a PEP 621 `pyproject.toml` and removed all metadata from `setup.py`. Builds now use the PEP 517 backend (`setuptools.build_meta`). If necessary for legacy workflows, I can keep a tiny shim `setup.py` that just calls `setuptools.setup()` with no metadata.

***2) Having pyproject.toml also allows more settings to be included there, instead of having separate files:
pytest.ini
.coveragerc
.flake8***

**Action taken** — I consolidated tool configuration into `pyproject.toml`:

* `pytest` under `[tool.pytest.ini_options]` (including `testpaths`, `addopts`, and targeted `filterwarnings` for docs/CI),
* coverage under `[tool.coverage.run]` / `[tool.coverage.report]`,
* formatting under `[tool.black]`,
* linting under `[tool.ruff]`.
  This removes the separate `pytest.ini`, `.coveragerc`, and `.flake8` files.

***3) try-except in setup.py for the kdiagram.**version** makes the package version redundant and unstable. Same in kdiagram/**init**.py, where it points to the *version.py file, but there, it will always run into an ImportError. Please unify this into optimally one place.***

**Action taken** — Version is now **single-sourced** in `pyproject.toml`. At runtime I read it via `importlib.metadata.version("k-diagram")`. When building from a source checkout without installation, I fall back to the same value as in `pyproject.toml` (no `_version.py`, no duplication, no import-time failure).

***4) Good having the most important functions exposed by the package in the **init**.py file. ??***

**Action taken** — Thanks! I kept the public API re-exports intact and only simplified version/warnings handling around them.

***5) You supress SyntaxWarnings globally in the package. While there might be several reasons to do this, I'd like to know your specific motivation in this case.***

**Action taken** — I removed all **global** suppression from the library. My earlier motivation was noisy `SyntaxWarning`s during Linux/Sphinx builds. I’ve replaced this with opt-in helpers (`configure_warnings`, `warnings_config`) and now apply narrow filters **only** in docs/tests (e.g., ignore `SyntaxWarning` from `numpy|matplotlib|pandas` during docs build). End-users get default Python behavior.

***6) The tests are inside the k-diagram package, but are not a subpackage, as the **init**.py is missing. "PytestConfigWarning: No files were found in testpaths"***

**Action taken** — I moved the test suite to a top-level `tests/` directory (not included in the install) and set `testpaths = ["tests"]` in `pyproject.toml`. Coverage targets `kdiagram`. This resolves discovery and keeps the layout conventional.

# CI

***1) Solve the two warnings in the current CI run.***
**Action taken** — The workflow was cleaned up and modernized:

* Upgraded to `actions/checkout@v4` and `conda-incubator/setup-miniconda@v3`.
* Ensured pytest always generates `coverage.xml` at the repo root and that the Codecov step points to it.
* Set `fail_ci_if_error: true` on the Codecov upload and pinned to `codecov/codecov-action@v4`.
  The warnings in the referenced run no longer appear in current runs.

***2) Coverage is measured/uploaded but not visible in README; only tests were covered.***
**Action taken** — Fixed the coverage target and visibility:

* Pytest now runs as `pytest -n auto --cov=kdiagram --cov-report=xml --cov-report=term-missing tests/`.
* Coverage config moved to `pyproject.toml` with `source = ["kdiagram"]` and `omit = ["kdiagram/__init__.py", ...]` so **library code** is measured (not `tests/`).
* Added a Codecov badge to `README.md` and enabled PR comments.
  Current local/CI coverage: **\~83.7%**.

***3) Consider running tests on more recent Python versions.***
**Action taken** — CI now tests a matrix of **3.9, 3.10, 3.11 and 3.12** (minimum + most recent supported), matching `requires-python=">=3.9"` and our Trove classifiers.

***4) flake8 produces 6k+ warnings; README says Black is used but formatting isn’t enforced.***
**Action taken** — Standardized and enforced a single toolchain:

* Replaced flake8 with **Black + Ruff** in CI.
* CI **fails** on style violations (`black --check .` and `ruff check .`).
* Centralized config in `pyproject.toml` (`[tool.black]` and `[tool.ruff]`), line length **88**, target versions **py39–py312**.
* Optional: Added `pre-commit` hooks locally to catch issues before push (documented in `CONTRIBUTING.rst`).

***5) Make the CI steps reproducible and minimal.***
**Action taken** — The workflow installs the package once, then runs lint + tests:

```yaml
- name: Install k-diagram (no deps; conda handles them)
  run: python -m pip install . --no-deps --force-reinstall

- name: Lint (Black + Ruff)
  run: |
    python -m pip install -e .[dev]
    black --check .
    ruff check .

- name: Test with pytest + coverage
  run: pytest -n auto --cov=kdiagram --cov-report=xml --cov-report=term-missing tests/

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v4
  with:
    token: ${{ secrets.CODECOV_TOKEN }}
    files: ./coverage.xml
    name: linux-py${{ matrix.python-version }}
    flags: unittests
    fail_ci_if_error: true
```

***6) Outcome.***
**Result** — CI now:

* Runs on **3.9 / 3.10 / 3.11/ 3.12**,
* Enforces **Black + Ruff** and fails on violations,
* Measures **package source** coverage correctly (now \~**83.7%**),
* Uploads to Codecov and shows a **badge** in the README,
* Avoids prior warnings and keeps logs clean.


