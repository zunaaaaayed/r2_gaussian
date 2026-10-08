# Week 3 report

Upload `TRDP2_Week3_Overleaf.zip` as a new Overleaf project. Select
`TRDP2_Week3.tex` as the main document and **pdfLaTeX** as the compiler. The archive
contains the TeX source, all three figures, this guide, and the frozen metrics.
No external bibliography file, shell escape, Python, or raw data is needed to compile.

`TRDP2_Week3.pdf` is the locally compiled five-page preview. Compile twice locally:

```bash
cd Reports/Week3
pdflatex -interaction=nonstopmode -halt-on-error TRDP2_Week3.tex
pdflatex -interaction=nonstopmode -halt-on-error TRDP2_Week3.tex
```

The report follows the five-section structure and author names of
`Reports/TRDP2_Week1.pdf` (the user's Week 2 report). The reporting window is
3–9 October 2026, with an explicit evidence cutoff of 8 October.

Figures:
- `tv_tradeoff.pdf`: PSNR, SSIM, and boundary MAE from the six frozen metric records.
- `slice_errors.png`: fixed central axis-2 slice, same reference and error windows.
- `convergence.pdf`: actual saved iteration evaluations, no smoothed/interpolated data points.

`make_figures.py` regenerates figures from the repository's frozen metrics and
local completed outputs; it requires the research Python environment, NumPy,
Matplotlib, and PyYAML. Large volumes are not included in the Overleaf archive.
`metrics_source.json` is a copy of the frozen result record used for the table
and trade-off figure. Reconstruction code and outputs are unchanged by this report.

Validation: twice compiled with pdfLaTeX, no unresolved references or layout
warnings in the final log; rendered pages inspected. All six table rows were
checked against the frozen results. Prior reports are preserved.
