.. _lab_installation:

============
Installation
============

This page explains how to install the ``k-diagram`` package. Choose
the method that best fits your workflow.

Requirements
------------

Before installing, ensure you have the following prerequisites:

* **Python:** version 3.9 or higher.
* **Core dependencies:** installed automatically when using
  ``pip``. The package relies on common scientific libraries:
  ``numpy``, ``pandas``, ``scipy``, ``matplotlib``, ``seaborn``,
  and ``scikit-learn``.

Install from PyPI (recommended)
-------------------------------

The easiest way to install ``k-diagram`` is via PyPI:

.. code-block:: bash

   pip install k-diagram

This installs the latest stable release together with all required
dependencies.

Upgrade to the newest version:

.. code-block:: bash

   pip install --upgrade k-diagram

Use a virtual environment
-------------------------

It is strongly recommended to install Python packages within a
virtual environment (using tools like ``venv`` or ``conda``). This
avoids conflicts between dependencies of different projects.

.. note::

   Using virtual environments keeps your global Python installation
   clean and ensures project dependencies are isolated.

**With ``venv`` (built-in):**

.. code-block:: bash

   # create env (here named .venv)
   python -m venv .venv

   # activate (Linux/macOS)
   source .venv/bin/activate
   # or on Windows (Command Prompt)
   # .venv\Scripts\activate.bat
   # or on Windows (PowerShell)
   # .venv\Scripts\Activate.ps1

   # install inside the env
   pip install k-diagram

   # deactivate when done
   # deactivate

**With ``conda``:**

.. code-block:: bash

   conda create -n kdiagram-env python=3.11
   conda activate kdiagram-env
   pip install k-diagram
   # conda deactivate

 
.. _development_install_source:

Development install (from source)
---------------------------------

If you want to contribute, run the latest source, or build docs,
install from the GitHub repository in *editable* mode.

1) Clone the repository:

.. code-block:: bash

   git clone https://github.com/earthai-tech/k-diagram.git
   cd k-diagram

2) Choose **one** of the following setups.

**A. Conda environment (reproducible toolchain)**

We provide an ``environment.yml`` that installs Python, runtime
deps, testing tools, linters, and the documentation toolchain.

.. code-block:: bash

   # create and activate the environment
   conda env create -f environment.yml
   conda activate k-diagram-dev

   # install the package (no extra deps; conda handled them)
   python -m pip install . --no-deps --force-reinstall

Notes:

* The environment name is ``k-diagram-dev`` (as defined in the
  file). If you prefer a different name, edit ``name:`` in
  ``environment.yml`` and use that name when activating.
* This path is ideal when you want a consistent setup that matches
  our CI configuration.

**B. Pure pip + editable install (no conda)**

If you prefer a lightweight setup using only ``pip``:

.. code-block:: bash

   # (optional) create and activate a venv first
   python -m venv .venv
   source .venv/bin/activate  # or Windows equivalent

   # install in editable mode with dev extras
   pip install -e .[dev]

The ``[dev]`` extra installs common development tools (pytest,
coverage, Ruff, Black, and Sphinx + extensions) defined in
``pyproject.toml``.

Verifying your installation
---------------------------

Open Python and import the package:

.. code-block:: python
   :linenos:

   import kdiagram
   print("k-diagram version:", getattr(kdiagram, "__version__", "unknown"))

If this runs without errors, your installation is working.

.. _building_documentation: 

Building Documentation
----------------------

After installing ``k-diagram`` (from PyPI or from source), you
can build the documentation locally with `Sphinx
<https://www.sphinx-doc.org/>`_ and the extensions listed in
``pyproject.toml``.

**1) Install documentation dependencies**

If you followed the editable :ref:`development_install_source`
with the ``[dev]`` extra, you’re all set. Otherwise, 
install the doc tools:

.. code-block:: bash

   pip install -e .[dev]

Or (if you prefer to keep testing/linting tools out of your
environment) use the docs requirements file:

.. code-block:: bash

   pip install -r docs/requirements.txt

**2) Build the HTML site**

Using the Makefile (created by ``sphinx-quickstart``):

.. code-block:: bash

   cd docs
   make html

Open ``docs/_build/html/index.html`` in your browser.

Alternatively, call ``sphinx-build`` directly (handy for CI or custom
builders):

.. code-block:: bash

   # If your conf.py is in docs/
   sphinx-build -b dirhtml docs docs/_build/html

   # If your conf.py is in docs/source/
   sphinx-build -b dirhtml docs/source docs/_build/html

The ``dirhtml`` builder produces “pretty” URLs (one folder per page).

**3) Clean builds (optional)**

Force a fresh build by removing the build directory first:

.. code-block:: bash

   rm -rf docs/_build && sphinx-build -b dirhtml docs docs/_build/html

Or with Make:

.. code-block:: bash

   cd docs
   make clean
   make html

.. note::

   On Windows, use ``.\make.bat html`` (and ``.\make.bat clean``)
   instead of ``make html``.

**4) Build PDF (optional)**

Requires a LaTeX distribution (TeX Live on Linux/macOS, MiKTeX on
Windows):

.. code-block:: bash

   cd docs
   make latexpdf

The PDF is written to ``_build/latex/k-diagram.pdf``.

**5) Recommended checks**

For link checking and warnings-as-errors during local QA:

.. code-block:: bash

   # treat warnings as errors (+ nitpicky mode)
   sphinx-build -nW -b dirhtml docs docs/_build/html

   # check external links (can be slow)
   make linkcheck

**Notes**

- If math doesn’t render, ensure your MathJax (or the offline
  plugin) is installed per your ``pyproject.toml`` extras.
- If citations don’t appear, confirm ``sphinxcontrib-bibtex`` is
  installed and that your ``conf.py`` includes the bibtex config.


Troubleshooting
---------------

* Ensure your ``pip`` is up to date:

  .. code-block:: bash

     pip install --upgrade pip

* If you build from source and a dependency needs compilation,
  make sure you have a working compiler toolchain appropriate for
  your OS.
* If you used ``conda`` and encounter solver conflicts, try
  updating ``conda`` and recreating the environment:

  .. code-block:: bash

     conda update -n base -c defaults conda
     conda env remove -n k-diagram-dev
     conda env create -f environment.yml

* Still stuck? Please open an issue with details about your OS,
  Python version, and the full error message:

  https://github.com/earthai-tech/k-diagram/issues
