The accompanying codes reproduce all figures and statistics presented in
"Cumulative differences between subpopulations versus body mass index in the
Behavioral Risk Factor Surveillance Survey data." This repository also provides
LaTeX and BibTeX sources for replicating the paper.

The main files in the repository are the following:

``tex/paper.pdf``
PDF version of the paper

``tex/paper.tex``
LaTeX source for the paper

``tex/paper.bib``
BibTeX source for the paper

``tex/fairmeta.cls``
LaTeX class for the paper

``tex/logometa.pdf``
PDF logo for Meta

``tex/figures/diffs0.fig``
Xfig file for odd centered differences of responses

``tex/figures/diffs0.pdf``
PDF file for odd centered differences of responses

``tex/figures/diffs1.fig``
Xfig file for even centered differences of responses

``tex/figures/diffs1.pdf``
PDF file for even centered differences of responses

``tex/figures/sums0.fig``
Xfig file for odd sums of weights

``tex/figures/sums0.pdf``
PDF file for odd sums of weights

``tex/figures/sums1.fig``
Xfig file for even sums of weights

``tex/figures/sums1.pdf``
PDF file for even sums of weights

``tex/figures/partition.fig``
Xfig file for partitioning the real line

``tex/figures/partition.pdf``
PDF file for partitioning the real line

``codes/disjoint.py``
Plots of the differences between two subpopulations with disjoint scores

``codes/disjoint_weighted.py``
Plots of weighted differences between two subpopulations with disjoint scores

``codes/paired_weighted.py``
Plots of cumulative differences between paired samples' responses, with weights

``codes/subpop_weighted.py``
Plots of deviation of a subpop. from the full pop., with weighted sampling

``codes/dists.py``
Calculate cumulative distribution functions for standard Brownian motions

``codes/brfss.py``
Performs a data analysis of the Behavioral Risk Factor Surveillance System

``codes/tm.py``
Performs a data analysis of the data set from Taylor and Mickel about CA's DDS

``codes/xlsx2csv.py``
Python script and library for converting Microsoft Excel files to CSV format
(re-distributed from Dilshod Temirkhodjaev under the MIT License)

The unit tests invoke [ImageMagick](https://imagemagick.org)'s ``convert``.

********************************************************************************

License

This cumbiostats software is licensed under the LICENSE file (the MIT license)
in the root directory of this source tree.
