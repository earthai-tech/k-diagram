Dear Reviewer Cubeth ,

Thank you very much for your thoughtful and constructive feedback on the `k-diagram` package. I truly appreciate 
your positive comments on the documentation structure and your excellent suggestion to integrate scientific 
references for the visualization methods. This is a crucial point that significantly enhances the scientific 
rigor and context of the project, and I am grateful you brought it to my attention.

You are absolutely right that readers would benefit from understanding the origins of each diagnostic plot—whether 
they are established methods or novel contributions from this work. I have now completed a comprehensive update to 
address this across the JOSS paper, the documentation, and the code's docstrings.

Here is a summary of the actions taken:

1.  **JOSS Paper (`paper.md`)**:
    * The **Functionality** section has been revised to provide clear scientific context. I have added citations 
    for established, foundational methods like Taylor Diagrams `[@Taylor2001]`, Reliability Diagrams `[@Jolliffe2012]`, 
    and Kernel Density Estimation `[@Silverman1986]`.
    * For the novel visualizations developed as part of this research (e.g., polar error bands, quiver plots, feature 
    fingerprints), I have cited our accompanying research paper `[@kouadiob2025]` to properly attribute their origin.

2.  **Full Documentation (User Guide & Docstrings)**:
    * I have integrated the `sphinxcontrib-bibtex` extension into the documentation build process to handle citations
    professionally. A central `references.bib` file has been created.
    * The **User Guide** pages (`uncertainty.rst`, `errors.rst`, `comparison.rst`, etc.) have been updated with `:cite:` 
    directives, linking the descriptions of the plots directly to their foundational papers.
    * Following your suggestion, the **docstrings** for the relevant functions have also been updated to include 
    a `References` section with plain-text citations, making the scientific context accessible directly from the code.
    * I also added an admonition box at the top of the uncertainty user guide, which directs readers to our research 
    paper for a practical, real-world case study of these plots in action.

About the abstract (Summary), I have moved the citation in the plain text. 

I believe these changes fully address your comments and make the package and its documentation much stronger. 
Thank you once again for your time and for providing such valuable guidance. Your feedback has been instrumental 
in improving the quality of this project.

Best regards,

