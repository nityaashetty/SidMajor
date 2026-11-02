# Report: IJCRSEE format

The SANF-Net report laid out to match the IJCRSEE template (`../IJCRSEEtemplate (1).doc`):
- A4, 2.5 cm margins, single spacing, justified text
- Helvetica Narrow standing in for Arial Narrow: 16 pt title, 10 pt abstract and keywords, 12 pt body
- APA (author-year) citations, at most three heading levels

## Build

`tectonic main.tex`, or upload this folder to Overleaf (pdfLaTeX + BibTeX).

## Files

| File | Section |
|---|---|
| `main.tex` | title block, page layout, includes |
| `sections/00_abstract.tex` | Abstract (≤250 words, no abbreviations) + keywords |
| `sections/01_introduction.tex` | Introduction, objectives, contributions |
| `sections/02_literature_survey.tex` | Literature survey (7-paper table + related work) |
| `sections/03_materials_methods.tex` | Data collection, preprocessing, biomarkers, evaluation protocol |
| `sections/04_baselines.tex` | Baseline models and results |
| `sections/05_proposed_model.tex` | SANF-Net architecture and training |
| `sections/06_ensemble_results.tex` | Final ensemble, test / CV results, ablations, confounds, external |
| `sections/07_dashboard.tex` | Clinical dashboard (not included yet) |
| `sections/08_discussion.tex` | Discussions + limitations |
| `sections/09_conclusions.tex` | Conclusions |
| `sections/10_declarations.tex` | Acknowledgements, funding, conflicts, data availability, IRB, author contributions |
| `references.bib` | copied from `../Paper/references.bib` |
| `figures/` | copied from `../Paper/figures/` |

## Where the numbers come from

- **Official split and ensemble (72.9%):** `../Approach11_StateAwareNormative/artifacts/runs/sanf_net/results.json`. This holds three seeds; the ensemble averages their probabilities.
- **Everything else** (CV, baselines, ablations, confounds, external): `../Paper/main.tex`, which is reproducible from `../Paper/analysis/`.

## Word version

`python docx_build/build_docx.py [output.docx]` builds the Word file on the journal template (default
`SANF-Net_IJCRSEE.docx`). Run `python docx_build/make_figures.py` first if the plots change.

## Optional additions

- Dashboard: `sections/07_dashboard.tex` holds an outline and is left out of the build. Once the dashboard exists, write
  it in and add it back in `main.tex` and in `SECTIONS` in `docx_build/build_docx.py`.
- UDC, doi and the revision and acceptance dates are assigned by the journal on acceptance.
