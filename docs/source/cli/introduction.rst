.. _cli_introduction:

====================
Introduction - CLI
====================

Ever feel like your standard charts are missing the full story?
Welcome to ``k-diagram``—a powerful suite of command-line tools
designed to give you a fresh, insightful perspective on your models
and data. We leverage the power of **polar coordinates** to turn
complex diagnostics into beautiful, intuitive visualizations. Think of
it as a new set of lenses for spotting patterns in performance,
uncertainty, and feature relationships that you might otherwise miss.

This page is your starting point. We'll cover the core concepts that
apply to all commands and then give you a guided tour to help you find
the perfect plot for your task.

.. tip::
   
   **A Note on Command Naming**
   
   Throughout this documentation, you will see the main command written
   as ``k-diagram`` (with a hyphen). However, a convenient alias
   ``kdiagram`` (without a hyphen) is also configured.

   Feel free to use whichever you prefer—they are completely
   interchangeable! For example, the following two commands are
   identical:

   .. code-block:: bash

      k-diagram plot-time-series data.csv --savefig plot.png

      kdiagram plot-time-series data.csv --savefig plot.png
      
      
------------------------------------------
The Core Philosophy: A Shared Grammar
------------------------------------------

The best part of the ``k-diagram`` CLI is that once you learn a few
simple patterns, you've learned the whole suite. Most commands share a
common "grammar" for handling data, selecting columns, and styling plots.

A typical command is as simple as this:

.. code-block:: bash

   kdiagram <COMMAND> your_data.csv --flag VALUE --savefig my_plot.png

**Input Data**
^^^^^^^^^^^^^^
All commands work directly with your tabular data files. Just provide
the path to your ``.csv`` or ``.parquet`` file as the first argument.
The format is detected automatically, but you can always override it
with ``--format``.

**Selecting Columns**
^^^^^^^^^^^^^^^^^^^^^
You'll often need to tell a command which columns to plot. We provide 
a few flexible ways to do this:

- For quick comparisons, you can repeat the ``--pred`` flag:

  .. code-block:: bash

     # Quickly plot two prediction columns
     --pred model_a_preds --pred model_b_preds

- For more structured plots, the named ``--model`` flag is clearer,
  especially when you have many models:

  .. code-block:: bash

     # Name your models for a clean legend
     --model "Linear Model":lm_preds --model "Tree Model":tree_preds

**Customizing & Saving Your Plots**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Every plot can be customized and saved with a shared set of intuitive
flags. By default, plots are shown in an interactive window, but you
can easily save them for reports or presentations.

While each command has unique options, most respond to a common set of
styling flags. A typical synopsis for these shared options looks like
this:

.. code-block:: text

   # General Appearance
   --title "My Plot Title"
   --figsize 10,8
   --cmap viridis

   # Scatter Plot Specifics (where applicable)
   [--s 50] [--alpha 0.7] [--marker "o"]

   # Grid and Axis Toggles
   --show-grid | --no-show-grid
   [--mask-angle | --no-mask-angle]
   [--mask-radius | --no-mask-radius]

   # Saving to a File
   --savefig my_figure.png
   [--dpi 300]

Now that you know the basic grammar, let's explore what you can build
with it.

---------------------------------------------------
Find the Right Tool: A Tour of the Commands
---------------------------------------------------

The commands are organized into thematic groups based on the questions
they help you answer. Many of these visualizations are rooted in
specific statistical concepts like forecast verification, calibration,
and error analysis. For a deeper dive into the theory behind the
plots, please refer to our detailed :doc:`../user_guide/index`.

Where would you like to begin?

**General & Contextual Plots**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Start here.** These are your essential, first-look tools for
visualizing raw time series, checking correlations, and getting a
baseline understanding of your model's errors. Effective visualization
is the cornerstone of data analysis :footcite:p:`Hunter:2007`.

- **CLI Reference**: :doc:`context/`
- **User Guide**: :doc:`../user_guide/context`
- **Examples Gallery**: :doc:`../gallery/context`

**Model Evaluation**
^^^^^^^^^^^^^^^^^^^^
**Ready to see which model wins?** These plots go beyond a single
score, offering classic evaluation metrics like ROC/PR curves
:footcite:p:`Powers2011`, confusion matrices, and the famous Taylor
diagram :footcite:p:`Taylor2001` for a holistic performance summary.

- **CLI Reference**: :doc:`evaluation/` and :doc:`taylor_diagram/`
- **User Guide**: :doc:`../user_guide/evaluation` and :doc:`../user_guide/taylor_diagram`
- **Examples Gallery**: :doc:`../gallery/evaluation` and :doc:`../gallery/taylor_diagram`

**Comparison & Calibration**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**How trustworthy are your model's predictions?** This group includes
tools like reliability diagrams to check if your forecast probabilities
are well-calibrated, alongside radar charts for direct,
multi-metric model comparisons.

- **CLI Reference**: :doc:`comparison/`
- **User Guide**: :doc:`../user_guide/comparison`
- **Examples Gallery**: :doc:`../gallery/comparison`

**Probabilistic Forecast Diagnostics**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**When a single number isn't enough.** A great probabilistic forecast
is both reliable (calibrated) and precise (sharp)
:footcite:p:`Gneiting2007b`. These advanced tools let you check if your
model's uncertainty estimates are actually trustworthy using methods
like PIT histograms and CRPS comparisons :footcite:p:`Jolliffe2012`.

- **CLI Reference**: :doc:`probabilistic/`
- **User Guide**: :doc:`../user_guide/probabilistic`
- **Examples Gallery**: :doc:`../gallery/probabilistic`

**Uncertainty Analysis**
^^^^^^^^^^^^^^^^^^^^^^^^
**How does your model's uncertainty behave?** Does it drift over time?
Are you capturing the outcomes you expect? These commands are dedicated
to diagnosing the quality and characteristics of your prediction
intervals, a key feature of modern forecasting systems
:footcite:p:`Lim2021, kouadiob2025`.

- **CLI Reference**: :doc:`uncertainty`
- **User Guide**: :doc:`../user_guide/uncertainty`
- **Examples Gallery**: :doc:`../gallery/uncertainty`

**Relationship & Error Analysis**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Dive deep into the mistakes.** A truly "good" forecast requires a
thorough understanding of its errors :footcite:p:`Murphy1993What`.
These plots help you uncover hidden biases and systematic patterns by
exploring the relationships between your model's errors, its
predictions, and the true values.

- **CLI Reference**: :doc:`relationship/` and :doc:`errors/`
- **User Guide**: :doc:`../user_guide/relationship` and :doc:`../user_guide/errors`
- **Examples Gallery**: :doc:`../gallery/relationship` and :doc:`../gallery/errors`

**Feature-Based Visualization**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Look beyond predictions to the features themselves.** Organizing your
data effectively is crucial :footcite:p:`Wickham2014`. These commands
leverage that structure to help you understand which features are most
important with "fingerprint" charts and how different features
interact to influence the outcome.

- **CLI Reference**: :doc:`feature_based/`
- **User Guide**: :doc:`../user_guide/feature_based`
- **Examples Gallery**: :doc:`../gallery/feature_based`

-------------------
Ready to Dive In?
-------------------

You now have a map of the entire ``k-diagram`` CLI. The best way to
learn is to try one out! Pick a section that matches your current task
and explore the commands within.

.. tip::
   Don't forget, you can get a full list of options and detailed help
   for any command by running it with the ``-h`` or ``--help`` flag.

   .. code-block:: bash

      kdiagram plot-time-series --help
      
.. raw:: html

    <hr>
    
.. rubric:: References

.. footbibliography::