.. _citing:

===================
Citing k-diagram
===================

If you use `k-diagram` in your research or work, please consider
citing it. Proper citation helps acknowledge the effort involved in
developing and maintaining this software and allows others to find
and verify the tools used.

We recommend citing both the software package itself and any relevant
publications describing the methods or applications.

Citing the Software
---------------------

For citing the `k-diagram` software package directly, please include
the author, title, version used, and the software repository URL.

**Recommended Format:**

   Kouadio, K. L. (2024). *k-diagram: Rethinking Forecasting
   Uncertainty via Polar-Based Visualization* (Version |release|).
   GitHub Repository. https://github.com/earthai-tech/k-diagram

*(Please replace |release| with the specific version of k-diagram you used
in your work. You can check the installed version using
``k-diagram --version`` or ``import kdiagram; print(kdiagram.__version__)``.)*

.. note::
   We plan to archive stable releases on platforms like Zenodo to provide
   a persistent Digital Object Identifier (DOI) for easier citation in
   the future. Please check the repository for updates on DOIs.

Related Publications
-----------------------

If your work relates to the concepts or applications demonstrated using
`k-diagram`, please also consider citing the relevant papers:

Application in Land Subsidence Forecasting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This paper introduces the application of the visualization techniques
within `k-diagram` in the context of an uncertainty-aware deep
learning framework for land subsidence forecasting.

* **Status:** Submitted for review (as of April 2025)
* **Title:** Understanding Uncertainty in Land Subsidence Forecasting
* **Authors:** Kouao Laurent Kouadio, Jianxi Liu, Kouamé Gbèlè Hermann
    Loukou, Liu Rong
* **Journal:** Submitted to *International Journal of Forecasting*

Software Overview Paper (Planned)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This upcoming paper will focus specifically on the `k-diagram` software
package, detailing its design, features, implementation, and usage.

* **Status:** In preparation (as of April 2025)
* **Tentative Title:** k-diagram: Rethinking Forecasting Uncertainty via
    Polar-Based Visualization
* **Author:** Kouao Laurent Kouadio
* **Target Journal:** Planned for submission to a relevant open-source
    software journal (e.g., *Journal of Open Source Software*).

BibTeX Entries
-----------------

For convenience, you can use the following BibTeX entries. Please update
fields like `year`, `version`, `doi`, `volume`, `pages` as appropriate
when the information becomes available or for the specific version you used.

**Citing the Software:**

.. code-block:: bibtex

   @software{kdiagram_software_2025,
     author       = {Kouadio, Kouao Laurent},
     title        = {{k-diagram: Rethinking Forecasting Uncertainty via Polar-Based Visualization}},
     version      = {|release|}, -- Add specific version used
     year         = {2024}, -- Or year of specific version release
     publisher    = {GitHub},
     url          = {https://github.com/earthai-tech/k-diagram},
     -- doi        = {10.5281/zenodo.XXXXXXX} -- Optional: Add Zenodo DOI if available
   }

**Citing the Land Subsidence Application Paper:**

.. code-block:: bibtex

   @unpublished{kouadio_subsidence_2025,
     author       = {Kouadio, Kouao Laurent and Liu, Jianxi and Loukou, Kouamé Gbèlè Hermann and Rong, Liu},
     title        = {{Understanding Uncertainty in Land Subsidence Forecasting}},
     note         = {Submitted to International Journal of Forecasting},
     year         = {2024}, -- Year of submission/potential publication
     -- url          = {https://arxiv.org/abs/xxxx.xxxxx} -- Optional: Add arXiv link if available
     -- doi        = {xx.xxxx/journal.xxxxxx} -- Optional: Add DOI when published
   }

**Citing the Planned Software Paper:**

.. code-block:: bibtex

   @misc{kouadio_kdiagram_paper_prep,
     author       = {Kouadio, Kouao Laurent},
     title        = {{k-diagram: Rethinking Forecasting Uncertainty via Polar-Based Visualization}},
     note         = {In preparation for submission},
     year         = {2025}, -- Expected year or TBD
     -- url          = {https://github.com/earthai-tech/k-diagram} -- Placeholder URL
   }

Thank you for citing `k-diagram`!