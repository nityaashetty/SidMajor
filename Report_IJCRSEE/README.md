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
| `sections/07_dashboard.tex` | **Placeholder**: clinical dashboard |
| `sections/08_discussion.tex` | Discussions + limitations |
| `sections/09_conclusions.tex` | Conclusions |
| `sections/10_declarations.tex` | Acknowledgements, funding, conflicts, data availability, IRB, author contributions |
| `references.bib` | copied from `../Paper/references.bib` |
| `figures/` | copied from `../Paper/figures/` |

## Where the numbers come from

- **Official split and ensemble (72.9%):** `../Approach11_StateAwareNormative/artifacts/runs/sanf_net/results.json`. This holds three seeds; the ensemble averages their probabilities.
- **Everything else** (CV, baselines, ablations, confounds, external): `../Paper/main.tex`, which is reproducible from `../Paper/analysis/`.

## Still to do (search for `todo`)

- Dashboard section and screenshot
- Full APA references for the six non-CAUEEG studies in the literature survey table, and the CEFA-Net authors
- Repository URL
- Author contribution roles
- Corresponding author: currently set to Siddarth Potdar; confirm
- UDC and doi are assigned by the journal
