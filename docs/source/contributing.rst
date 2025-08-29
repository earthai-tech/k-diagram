.. _contributing:

===============
Contributing
===============

We welcome contributions to `k-diagram`! Whether you're fixing a bug,
adding a new feature, improving documentation, or suggesting ideas,
your help is valued. Thank you for your interest in making `k-diagram`
better.

Getting Started
---------------

* **Issues Tracker:** The best place to start is the
  `GitHub Issues page <https://github.com/earthai-tech/k-diagram/issues>`_.
  Look for existing issues labeled `bug`, `enhancement`,
  `documentation`, or `good first issue`.
* **Ideas:** If you have an idea for a new feature or improvement,
  feel free to open a new issue to discuss it first.
* **Questions:** If you have questions about usage or contribution,
  you can also use the `GitHub Issues <https://github.com/earthai-tech/k-diagram/issues>`_
  page.

Setting up for Development
-----------------------------

To make changes to the code or documentation, you'll need to set up
a development environment. Please follow the instructions in the
:ref:`installation guide <lab_installation>` under the section
**"Installation from Source (for Development)"**. This typically
involves:

1.  Forking the repository on GitHub.
2.  Cloning your fork locally (`git clone ...`).
3.  Installing the package in editable mode with development
    dependencies (`pip install -e .[dev]`). Using a virtual environment
    is highly recommended.

Making Changes
------------------

1.  **Create a Branch:** Create a new branch from the `main` branch
    (or the current development branch) for your changes. Use a
    descriptive name (e.g., `fix/plot-legend-overlap` or
    `feature/add-confidence-bands`).

    .. code-block:: bash

       git checkout main
       git pull upstream main # Keep your main branch updated
       git checkout -b your-descriptive-branch-name

2.  **Code Style:** Please follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_
    guidelines and strive for code consistency with the existing
    codebase. Use linters like Flake8 if possible.

3.  **Docstrings:** Write clear and informative docstrings for any new
    functions or classes, following the NumPy docstring standard
    (as used throughout the project). Ensure existing docstrings are
    updated if function signatures or behavior change.

4.  **Testing:** `k-diagram` uses `pytest` for testing.
    * Add new tests for any new features you implement.
    * Add or update tests to cover any bug fixes.
    * Ensure all tests pass before submitting your changes. Run tests 
    from the project root directory:

    .. code-block:: bash

       pytest tests/ # Or simply 'pytest'

5.  **Documentation:** If your changes affect the user interface, add
    new features, or change behavior, please update the relevant
    documentation files (in `docs/source/`). Build the documentation
    locally to check formatting:

    .. code-block:: bash

       # Navigate to the docs directory
       cd docs
       # Build the HTML documentation
       make html
       # Open _build/html/index.html in your browser

6.  **Commit Changes:** Make clear, concise commit messages.

Submitting a Pull Request
----------------------------

1.  **Push to Fork:** Push your changes to your forked repository on
    GitHub:

    .. code-block:: bash

       git push origin develop # [ develop as descriptive-branch-name]

2.  **Open Pull Request:** Go to the original `k-diagram` repository
    on GitHub (`earthai-tech/k-diagram`) and open a Pull Request (PR)
    from your branch to the `k-diagram` `main` branch.

3.  **Describe PR:** Write a clear description of the changes you made
    and why. Link to the relevant GitHub issue(s) using `#issue-number`.

4.  **Checks:** Ensure any automated checks (Continuous Integration,
    linters) configured for the repository pass on your PR.

5.  **Review:** Your PR will be reviewed by the maintainers. Be
    prepared to discuss your changes and make adjustments based on
    feedback.

Code of Conduct
---------------

All participants in the `k-diagram` project (contributors,
maintainers, users in community spaces) are expected to adhere to
the project's :doc:`Code of Conduct <code_of_conduct>`. Please review
this document to understand the expected standards of behavior. We
strive to foster an open, welcoming, and respectful community.


Thank you again for your contribution!
