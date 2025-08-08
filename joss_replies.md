## Documentation 

**1. A11y (accessibility) hover contrast**
Thank you for pointing out the low-contrast hover state in the sidebar. I’ve updated my `custom.css` so that 
hovered and active links use our white background color (`--color-background-primary`) instead of black:

```css
.sidebar-drawer .reference:hover {
  background-image: none !important;               
  background-color: var(--color-brand-primary) !important;
  color: var(--color-background-primary) !important;
  padding-left: 1.1em;
}
.sidebar-drawer li.current > a.reference {
  background-image: none !important;
  background-color: var(--color-brand-primary) !important;
  color: var(--color-foreground-primary) !important;
  font-weight: 600;
}
```
Since there is no gradient, it now shows a better constrast. Further 
I rebuilt the docs and confirmed these hover states now meet WCAG contrast guidelines.

---

**2. Proper NumPy-style docstring parsing**
You’re correct that Napoleon alone doesn’t enable NumPy’s full suite of directives. I’ve now installed 
and enabled the official **numpydoc** extension in `conf.py`:

```diff
 extensions = [
+   'numpydoc',                # NumPy’s own parser & directives
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    …
 ]
```

With `numpydoc` active (alongside my existing Napoleon settings), the autosummary 
pages—including `plot_anomaly_magnitude`—now render Parameters, Notes, Examples, and math 
blocks correctly. I’ve also consolidated duplicate **Notes** sections and fixed indentation in 
the affected docstrings.

