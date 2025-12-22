.. _citing:

===================
Citing k-diagram
===================

If you use ``k-diagram`` in your research or work, please consider
citing it. Proper citation acknowledges the effort involved in
developing and maintaining the software and helps others find and
verify the tools you used.

We recommend citing the software paper (JOSS) **and** (optionally) the
software package/repository version you used. You may also cite any
related methods or application papers that informed your work.

Software Paper (JOSS)
---------------------

This paper describes the ``k-diagram`` software and its core ideas.

.. code-block:: bibtex

    @article{Kouadio2025,
      doi       = {10.21105/joss.08661},
      url       = {https://doi.org/10.21105/joss.08661},
      year      = {2025},
      publisher = {The Open Journal},
      volume    = {10},
      number    = {116},
      pages     = {8661},
      author    = {Kouadio, Kouao Laurent},
      title     = {k-diagram: Rethinking Forecasting Uncertainty via Polar-based Visualization},
      journal   = {Journal of Open Source Software}
    }

Citing the Software Package
---------------------------

If you wish to cite the software artifact directly (e.g., a specific
version in a reproducible workflow), include the author, title, version
used, and repository URL.

**Recommended format:**

  Kouadio, K. L. (2025). *k-diagram: Rethinking Forecasting Uncertainty via
  Polar-based Visualization* (Version |release|). GitHub Repository.
  https://github.com/earthai-tech/k-diagram

.. note::

   Replace ``|release|`` with the specific version you used. You can
   check the installed version with ``k-diagram --version`` or
   ``import kdiagram; print(kdiagram.__version__)``.

Related Publications
--------------------

If your work uses concepts, diagnostics, or applications demonstrated
with ``k-diagram``, consider citing the relevant papers below.

.. note::

   Some entries are submitted. DOI, volume, and page
   information will be added once available.

CAS: Cluster-Aware Scoring for Probabilistic Forecasts (IJF submission)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This paper introduces **CAS (Cluster-Aware Severity / Cluster-Aware Scoring)**,
a metric that penalizes bursty, clustered forecast errors and complements
traditional scoring rules.

Examples and reproducible scripts are provided in the repository:
https://github.com/earthai-tech/k-diagram/tree/main/examples/cas

.. code-block:: bibtex

    @unpublished{kouadio_cas_ijf_2025,
      author  = {Kouadio, Kouao Laurent and Liu, Rong},
      title   = {CAS: Cluster-Aware Scoring for Probabilistic Forecasts},
      journal = {International Journal of Forecasting},
      note    = {Submitted},
      year    = {2025}
    }

Physics-Informed Urban Land Subsidence Forecasting (Nature submission)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This paper introduces GeoPriorSubsNet, a physics-informed deep learning
framework for probabilistic, city-scale land-subsidence forecasting and
diagnostics.

.. code-block:: bibtex

    @unpublished{kouadio_geopriorsubsnet_nature_2025,
      author  = {Kouadio, Kouao Laurent and Liu, Rong and Jiang, Shiyu and
                 Liu, Zhuo and Kouamelan, Serge and Liu, Wenxiang and
                 Qing, Zhanhui and Zheng, Zhiwen},
      title   = {Physics-Informed Deep Learning Reveals Divergent Urban Land Subsidence Regimes},
      journal = {Nature Communications},
      note    = {Submitted},
      year    = {2025}
    }

A Diagnostic Framework for Spatiotemporal Forecast Uncertainty (Environmental Modelling \& Software submission)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This paper proposes a diagnostic framework for interpreting probabilistic
forecast uncertainty and applies it to land subsidence forecasting,
including comparisons across model families.

.. code-block:: bibtex

    @unpublished{kouadio_diagnostic_framework_ems_2025,
      author  = {Kouadio, Kouao Laurent and Liu, Rong and Loukou, Kouam{\'e} Gb{\`e}l{\`e} Hermann and
                 Liu, Wenxiang and Qing, Zhanhui and Liu, Zhuo},
      title   = {A Diagnostic Framework for Spatiotemporal Forecast Uncertainty},
      journal = {Environmental Modelling \& Software},
      note    = {Submitted},
      year    = {2025}
    }

Thank you for citing ``k-diagram``!
