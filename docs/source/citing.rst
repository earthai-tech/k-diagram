.. _citing:

===================
Citing k-diagram
===================

If you use ``k-diagram`` in your research or work, please consider
citing it. Proper citation acknowledges the effort involved in
developing and maintaining the software and helps others find and
verify the tools you used.

We recommend citing the software paper (JOSS) **and** the software
package itself. You may also cite any related application or methods
papers that informed your work.

Software Paper (Planned JOSS Submission)
----------------------------------------

This paper focuses on the ``k-diagram`` software and is intended for
the open-source and scientific software community.

.. code-block:: bibtex

    @article{kouadio_kdiagram_joss_prep_2025,
      author       = {Kouadio, Kouao Laurent},
      title        = {k-diagram: Rethinking Forecasting Uncertainty via
                      Polar-Based Visualization},
      note         = {In preparation for submission to the Journal of
                      Open Source Software (JOSS)},
      year         = {2025},
      howpublished = {\url{https://github.com/earthai-tech/k-diagram}},
      release      = {|release|}
    }

Citing the Software Package
---------------------------

If you wish to cite the software artifact directly, include the
author, title, version used, and repository URL.

**Recommended format:**

  Kouadio, K. L. (2025). *k-diagram: Rethinking Forecasting
  Uncertainty via Polar-Based Visualization* (Version |release|).
  GitHub Repository. https://github.com/earthai-tech/k-diagram


.. note::
   
   Replace ``|release|`` with the specific version you used. You can
   check the installed version with ``k-diagram --version`` or
   ``import kdiagram; print(kdiagram.__version__)``.)

   Furthermore, we plan to archive stable releases on Zenodo to provide a persistent
   DOI. Please check the repository for updates once DOIs are issued.

Related Publications
--------------------

If your work uses concepts, diagnostics, or applications demonstrated
with ``k-diagram``, consider citing the relevant papers below.

.. note::

   Some entries are submitted or in preparation. DOI, volume, and page
   information will be added once available.

Land Subsidence Uncertainty Analysis (IJF submission)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This paper analyzes the structure and consistency of predictive
uncertainty in land-subsidence forecasting using diagnostic diagrams
related to ``k-diagram``.

.. code-block:: bibtex

    @unpublished{kouadio_subsidence_ijf_2025,
      author  = {Kouadio, Kouao Laurent and Liu, Rong and
                 Loukou, Kouam{\\'e} Gb{\\`e}l{\\`e} Hermann},
      title   = {Analytics Framework for Interpreting Spatiotemporal
                 Probabilistic Forecasts},
      journal = {International Journal of Forecasting},
      note    = {Submitted},
      year    = {2025}
    }

Urban Land Subsidence Forecasting (Nature Communications)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This paper introduces a physics-informed deep learning framework and
applies visualization techniques related to ``k-diagram`` to forecast
urban land subsidence.

.. code-block:: bibtex

    @article{liu_subsidence_nat_comm_2025,
      author  = {Kouadio, Kouao Laurent and Liu, Rong and Jiang, Shiyu
                 and Liu, Jianxin and Kouamelan, Serge and Liu, Wenxiang
                 and Qing, Zhanhui and Zheng, Zhiwen},
      title   = {Forecasting Urban Land Subsidence in the Era of Rapid
                 Urbanization and Climate Stress},
      journal = {Nature Communications},
      year    = {2025},
      note    = {Submitted}
    }

Thank you for citing ``k-diagram``!
