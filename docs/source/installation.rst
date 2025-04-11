.. _installation:

============
Installation
============

This page guides you through installing the `k-diagram` package.
Choose the method that best suits your needs.

Requirements
------------

Before installing, ensure you have the following prerequisites:

* **Python:** Version 3.6 or higher.
* **Core Dependencies:** `k-diagram` relies on several standard
    scientific Python libraries:
    * ``numpy``
    * ``pandas``
    * ``scipy``
    * ``matplotlib``
    * ``seaborn``
    * ``scikit-learn``

    These dependencies are typically installed automatically when you
    install `k-diagram` using ``pip``.

Installation from PyPI (Recommended)
------------------------------------

The easiest and recommended way to install `k-diagram` is directly
from the Python Package Index (PyPI) using ``pip``:

.. code-block:: bash

   pip install k-diagram

This command downloads and installs the latest stable release of the
package along with its required dependencies.

Installation from Source (for Development)
------------------------------------------

If you want to contribute to `k-diagram`, modify the code, or use
the very latest (potentially unstable) version, you can install it
from the source repository on GitHub.

1.  **Clone the repository:**
    First, clone the repository to your local machine using Git:

    .. code-block:: bash

       git clone https://github.com/earthai-tech/k-diagram.git
       cd k-diagram

2.  **Install in editable mode:**
    Install the package in "editable" mode using ``pip``. This links
    the installed package directly to your cloned source code, so any
    changes you make are immediately reflected without reinstalling.

    It's also recommended to install the optional development
    dependencies (`[dev]`), which include tools for testing and
    building documentation:

    .. code-block:: bash

       pip install -e .[dev]

    * The ``-e`` flag stands for "editable".
    * The ``.`` refers to the current directory (the cloned repo root).
    * ``[dev]`` installs the extra dependencies listed under `"dev"`
        in ``setup.py`` (like ``pytest``, ``sphinx``).

Virtual Environments (Highly Recommended)
-----------------------------------------

It is strongly recommended to install Python packages within a
virtual environment (using tools like ``venv`` or ``conda``). This
avoids conflicts between dependencies of different projects.

.. note::

   Using virtual environments keeps your global Python installation
   clean and ensures project dependencies are isolated.

**Using `venv` (Python's built-in tool):**

.. code-block:: bash

   # Create a virtual environment (e.g., named .venv)
   python -m venv .venv

   # Activate it (Linux/macOS)
   source .venv/bin/activate
   # OR Activate it (Windows - Command Prompt)
   # .venv\Scripts\activate.bat
   # OR Activate it (Windows - PowerShell)
   # .venv\Scripts\Activate.ps1

   # Now install k-diagram inside the active environment
   pip install k-diagram

   # Deactivate when finished
   # deactivate

**Using `conda`:**

.. code-block:: bash

   # Create a new conda environment (e.g., named kdiagram-env)
   conda create -n kdiagram-env python=3.9 # Or your preferred Python version

   # Activate the environment
   conda activate kdiagram-env

   # Install k-diagram
   pip install k-diagram # Often best to use pip within conda for PyPI packages

   # Deactivate when finished
   # conda deactivate


Verifying the Installation
--------------------------

After installation, you can verify it by importing the package in a
Python interpreter or script:

.. code-block:: python
   :linenos:

   import kdiagram

   try:
       print(f"k-diagram version: {kdiagram.__version__}")
   except AttributeError:
       print("Could not determine k-diagram version.")

If this runs without errors, the installation was likely successful.

Troubleshooting
---------------

If you encounter issues during installation:

* Ensure you have a compatible version of Python installed and that
    ``pip`` is up-to-date (``pip install --upgrade pip``).
* Check that you have the necessary build tools if installing from
    source or if a dependency requires compilation.
* If you face persistent problems, please consult the project's
    `GitHub Issues <https://github.com/earthai-tech/k-diagram/issues>`_
    page. Search for similar issues or open a new one with details
    about your environment and the error message.