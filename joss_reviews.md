Dear Reviewer Cubeth ,

Thank you very much for your thoughtful and constructive feedback on the 
`k-diagram` package. I truly appreciate  your positive comments on 
the documentation structure and your excellent suggestion to integrate 
scientific references for the visualization methods. This is a crucial 
point that significantly enhances the scientific rigor and context 
of the project, and I am grateful you brought it to my attention.

You are absolutely right that readers would benefit from understanding 
the origins of each diagnostic plot—whether they are established methods 
or novel contributions from this work. I have now completed a comprehensive 
update to address this across the JOSS paper, the documentation, and the 
code's docstrings.

Here is a summary of the actions taken:

1.  **JOSS Paper (`paper.md`)**:

    * The **Functionality** section has been revised to provide clear 
    scientific context. I have added citations for established, foundational 
    methods like Taylor Diagrams `[@Taylor2001]`, Reliability Diagrams `[@Jolliffe2012]`, 
    and Kernel Density Estimation `[@Silverman1986]`.
    * For the novel visualizations developed as part of this research 
    (e.g., polar error bands, quiver plots, feature fingerprints), I have 
    cited our accompanying research paper `[@kouadiob2025]` to properly 
    attribute their origin.
    
    However it you want me to insert some equations, Im willing to do it so.

2.  **Full Documentation (User Guide & Docstrings)**:

    * I have integrated the `sphinxcontrib-bibtex` extension into the 
    documentation build process to handle citations professionally. 
    A central `references.bib` file has been created.
    * The **User Guide** pages (`uncertainty.rst`, `errors.rst`, 
    `comparison.rst`, etc.) have been updated with `:footcite:` directives( 
    ``footsize``  to avoid duplicated citations: https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html ), 
    linking the descriptions of the plots directly to their foundational papers.
    
    * Following your suggestion, the **docstrings** for the relevant functions 
    have also been updated to include a `References` section with plain-text 
    citations, making the scientific context accessible directly from the code.
    
    * I also added an admonition box at the top of the uncertainty user guide, 
    which directs readers to our research 
    paper for a practical, real-world case study of these plots in action.

About the abstract (Summary), I have moved the citation in the plain text. 

I believe these changes fully address your comments and make the package and 
its documentation much stronger. Thank you once again for your time and for 
providing such valuable guidance. Your feedback has been instrumental 
in improving the quality of this project.

Best regards,

# CODE 


## warnings issues 

Dear [Reviewer's GitHub Username or Name],

Thank you for your detailed follow-up and for pushing for more rigor 
regarding the test warnings. I appreciate your diligence. You were absolutely 
right to be concerned about the use of `filterwarnings` in `pyproject.toml`, 
as globally silencing warnings can indeed hide underlying issues.

I have taken your feedback seriously and have implemented a more robust and 
transparent approach. **I have now removed the `filterwarnings` from `pyproject.toml` entirely.**

Instead, I have addressed each category of warning directly within 
the test suite using `pytest.warns`, ensuring that we only ignore warnings 
that are known, expected, and harmless. Here is a summary of the changes:

1.  **`FigureCanvasAgg is non-interactive` Warning**: This warning is 
specific to running tests in a non-interactive environment. I have now 
wrapped the specific plotting tests that trigger this in a `pytest.warns` block. 
This confirms the behavior is expected in our CI environment without silencing
 other, potentially important `UserWarning`s.

2.  **`tight_layout` Warning**: As you noted, this is a known, harmless 
warning from Matplotlib for complex `gridspec` layouts. Rather than ignoring 
it globally, the specific tests that generate these plots 
(e.g., `test_quantile_wilson_counts_bottom_multi_model`) now explicitly 
catch this warning with `pytest.warns`. This documents that the warning is 
expected for that specific plot.

3.  **NumPy 2.0 Deprecation Warnings (e.g., `np.find_common_type`)**: Your 
point about these being important is well-taken. My strategy is to ensure 
`k-diagram` is ready for NumPy 2.0. To manage this transition, I have created 
a compatibility module (`kdiagram/compat/numpy.py`) to handle version 
differences. The `DeprecationWarning`s are now caught in the relevant 
tests with `pytest.warns`, which makes our test suite aware of them while 
we finalize the transition to the new NumPy API.

By moving these checks into the test suite, we are no longer hiding any 
warnings. Instead, we are explicitly acknowledging and validating them 
where they occur. This has made our test suite cleaner and more precise.

Thank you again for your invaluable feedback. It has led to a significant 
improvement in the quality and robustness of the project's testing practices.
