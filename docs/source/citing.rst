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

.. note::

   Please note that details for submitted or planned publications
   (like DOI, volume, pages) are placeholders. This information will be
   updated once the papers are formally published.
   
Land Subsidence Uncertainty Analysis (IJF Submission)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This paper delves into analyzing the structure and consistency of
predictive uncertainty in land subsidence forecasting, applying methods
related to `k-diagram`.

.. code-block:: bibtex

    @unpublished{kouadio_subsidence_ijf_2025,
      author       = {Kouadio, Kouao Laurent and Liu, Rong and Loukou, Kouamé Gbèlè Hermann and Liu, Rong}, 
      title        = {{Understanding Uncertainty in Land Subsidence Forecasting}},
      journal      = {International Journal of Forecasting},
      note         = {Submitted},
      year         = {2025}, 
    }
    
    
Urban Land Subsidence Forecasting (Nature Sustainability Submission)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This paper introduces the XTFT deep learning framework and applies
visualization techniques (related to those in `k-diagram`) to forecast
land subsidence in Nansha and Zhongshan, China.

.. code-block:: bibtex

    @article{liu_subsidence_nat_sust_2025,
      author       = {Liu, Rong and Jiang, Shiyu and Kouadio, Kouao Laurent and Liu, Jianxin and Liu, Wenxiang and Qing, Zhanhui and Zheng, Zhiwen},
      title        = {{Forecasting Urban Land Subsidence in the Era of Rapid Urbanization and Climate Stress}},
      journal      = {Nature Sustainability},
      year         = {2025}, 
      note         = {Submitted},
    }
    

Software Paper (Planned JOSS Submission)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This upcoming paper will provide a focused description of the `k-diagram`
software package itself, targeting the open-source software community.

.. code-block:: bibtex

    @misc{kouadio_kdiagram_joss_prep_2025,
      author       = {Kouadio, Kouao Laurent},
      title        = {{k-diagram: Rethinking Forecasting Uncertainty via Polar-Based Visualization}},
      note         = {In preparation for submission to Journal of Open Source Software},
      year         = {2025},
      howpublished = {\url{https://github.com/earthai-tech/k-diagram}},
      release      = |release|
    }


Thank you for citing `k-diagram`!
