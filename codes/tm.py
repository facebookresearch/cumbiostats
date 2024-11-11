#!/usr/bin/env python3

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Plot cumulative differences for Taylor's and Mickel's "Simpson's Paradox..."

This script requires at least one argument on the command line and allows up to
two more. The first argument must be either a gender or an ethnicity. "Gender"
means "Female" or "Male", while "ethnicity" means "Asian", "Black", "Hispanic",
or "White". When provided, the second argument must also be either a gender or
an ethnicity ... and the second argument must be distinct from the first.
Providing a second argument allows for inclusion of a third argument. When
provided, the third argument must be an integer. Providing only one argument
results in comparing the indicated subpopulation to the full population.
Providing two arguments results in comparing the two indicated subpopulations
directly, with their scores (namely, the ages of the individuals) randomly
perturbed slightly such that the scores become distinct from each other.
Providing three arguments results in comparing the two indicated subpopulations
directly again, but now using the third argument to seed the random number
generator used for perturbing the scores. Omitting the third argument defaults
to using 543216789 as the random seed.

This script plots reliability diagrams and cumulative differences between
subpopulations (where "subpopulation" can mean the full population, too). The
script also saves text files of numerical statistics associated with the plots.
The script creates either a directory, "unweighted", or a directory,
"unweighted[random_seed]", within the working directory, if the directories do
not already exist. If there is only a single argument specified on the command-
line, then the script works with the first directory, "unweighted". If there is
more than a single argument specified on the command-line, then the script
works with the second directory, "unweighted[random_seed]". The script then
creates a subdirectory whose name is the identical to the sole argument given
by the command-line argument, if there is only one. If there is more than just
a single command-line argument, then the script creates a subdirectory whose
name is the two adjectives from the command-line, separated by an underscore.
Creation of any of these subdirectories is contingent on the subdirectories not
already existing.

In the lowest-level subdirectories, the script creates reliability diagrams
with the numbers of bins specified by the list nin defined below. Reliability
diagrams whose bins are equispaced along the scores (where the scores are the
values of the covariate, age) get saved to files, "equiscores[nbin].pdf", where
"[nbin]" is the number of bins for each of the subpopulations being compared
(with one of the "subpopulations" being the full population when considering
the directory, "unweighted"). Reliability diagrams whose bins contain roughly
the same number of observations for every bin get saved to files,
"equierrs[nbin].pdf", where again "[nbin]" gives the number of bins. The script
also saves the plot of cumulative differences to a file, "cumulative.pdf", as
well as a file, "metrics.txt", reporting numerical statistics associated with
the plots and their corresponding data. The responses for the populations are
the total expenditures that California's Department of Developmental Services
made in a year for the individuals in the data set. The script will overwrite
any files that already exist.

The data analysis all concerns data from Stanley A. Taylor and Amy E. Mickel,
"Simpson's Paradox: a data set and discrimination case study exercise," Journal
of Statistics Education, 22(1): 2014, 1-18. The data is available at
`dataset <http://www.StatLit.org/XLS/2014-Taylor-Mickel-Paradox-Data.xlsx>`_;
`docs <http://jse.amstat.org/v22n1/mickel/paradox_documentation.docx>`_
provides documentation. The script starts by downloading the former (the data)
in Microsoft Excel format, if necessary, to a file, "taylor-mickel.xlsx". Then
the script converts the Microsoft Excel file, "taylor-mickel.xlsx", into a file
of comma-separated values, "taylor-mickel.csv", without deleting the original
Microsoft Excel file. The script preserves the file of comma-separated values,
"taylor-mickel.csv", if it already exists.

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import argparse
import numpy as np
import os
import urllib.request
from numpy.random import default_rng

import disjoint
import dists
import subpop_weighted
from xlsx2csv import Xlsx2csv


# Specify whether to randomize the scores in order to ensure their uniqueness,
# in the case of comparing a subpopulation to the full population; note that
# RANDOMIZE will be set to True when comparing two subpopulations directly.
RANDOMIZE = False

# Specify the name of the files for the data from Taylor's and Mickel's
# "Simpson's Paradox...."
filename = 'taylor-mickel'

# Download the data set if necessary.
filexlsx = filename + '.xlsx'
if not os.path.isfile(filexlsx):
    print('downloading data...')
    urlxlsx = 'http://www.StatLit.org/XLS/2014-Taylor-Mickel-Paradox-Data.xlsx'
    urllib.request.urlretrieve(urlxlsx, filexlsx)

# Convert the xlsx (Microsoft Excel) file to csv (comma-separated value) format
# if necessary.
filename += '.csv'
if not os.path.isfile(filename):
    print('converting from xlsx to csv...')
    sheetid = 1
    kwargs = {
        'delimiter': ',',
        'quoting': 0,
        'sheetdelimiter': '--------',
        'dateformat': None,
        'timeformat': None,
        'floatformat': None,
        'scifloat': False,
        'skip_empty_lines': False,
        'skip_trailing_columns': False,
        'escape_strings': False,
        'no_line_breaks': False,
        'hyperlinks': False,
        'include_sheet_pattern': '^.*$',
        'exclude_sheet_pattern': '',
        'exclude_hidden_sheets': False,
        'merge_cells': False,
        'outputencoding': 'utf-8',
        'lineterminator': '\n',
        'ignore_formats': [''],
        'skip_hidden_rows': True,
    }
    csv = Xlsx2csv(filexlsx, **kwargs)
    csv.convert(filename, sheetid)

# Parse the command-line arguments (if any).
parser = argparse.ArgumentParser()
parser.add_argument(
    'subpop',
    choices=['Female', 'Male', 'Asian', 'Black', 'Hispanic', 'White'])
parser.add_argument(
    'subpop2', nargs='?',
    choices=['Female', 'Male', 'Asian', 'Black', 'Hispanic', 'White'])
parser.add_argument(
    'seed_for_rng', nargs='?', default=543216789, type=int)
clargs = parser.parse_args()

# Determine the attribute used for selecting subpopulations.
if clargs.subpop in ['Female', 'Male']:
    attribute = 'Gender'
else:
    attribute = 'Ethnicity'

if clargs.subpop2 is not None:
    RANDOMIZE = True
    # Ensure that the first and second subpops are compatible.
    if clargs.subpop == 'Female':
        if clargs.subpop2 != 'Male':
            raise ValueError(
                'If subpop is "Female," then subpop2 must be "Male."')
    if clargs.subpop == 'Male':
        if clargs.subpop2 != 'Female':
            raise ValueError(
                'If subpop is "Male," then subpop2 must be "Female."')
    if clargs.subpop in ['Asian', 'Black', 'Hispanic', 'White']:
        if clargs.subpop2 not in ['Asian', 'Black', 'Hispanic', 'White']:
            raise ValueError(
                'If subpop is an ethnicity, '
                + 'then subpop2 must be another ethnicity.')
        if clargs.subpop2 == clargs.subpop:
            raise ValueError(
                'If subpop is an ethnicity, '
                + 'then subpop2 must be a distinct ethnicity.')

# Count the number of lines in the file for filename.
lines = 0
with open(filename, 'r') as f:
    for line in f:
        lines += 1
print(f'reading and filtering all {lines} lines from {filename}....')

# Determine the number of columns in the file for filename.
with open(filename, 'r') as f:
    line = f.readline()
    num_cols = line.count(',') + 1

# Read and store the third through sixth columns in the file for filename.
with open(filename, 'r') as f:
    for line_num, line in enumerate(f):
        if line[-1] == '\n':
            line = line[:-1]
        parsed = line.split(',')[2:6]
        if line_num == 0:
            # The initial line is a header ... save its column labels.
            header = parsed.copy()
        else:
            # All but the initial line consist of data ... save the data.
            data = []
            for s in parsed:
                if s.isdecimal():
                    # Convert a whole number to an int.
                    data.append(int(s))
                else:
                    # Save the entire string.
                    if s == 'White not Hispanic':
                        data.append('White')
                    else:
                        data.append(s)
            # Initialize raw if line_num == 1.
            if line_num == 1:
                raw = []
            # Record the data.
            raw.append(data)

print(f'len(raw) = {len(raw)}')

# Read the scores and responses.
s = np.asarray([data[header.index('Age')] for data in raw], dtype=np.float64)
r = np.asarray(
    [data[header.index('Expenditures')] for data in raw], dtype=np.float64)
if RANDOMIZE:
    # Dither the scores in order to ensure the uniqueness of the ages.
    rng = default_rng(seed=clargs.seed_for_rng)
    s0 = s.copy()
    s = s + rng.standard_normal(size=s.shape) * s.max() * 1e-8
    assert np.unique(s).size == s.size

# Sort according to the scores.
perm = np.argsort(s, kind='stable')
s = s[perm]
r = r[perm]
if RANDOMIZE:
    s0 = s0[perm]

# Determine the indices for the subpopulation(s).
if clargs.subpop2 is None:
    inds = [raw[ind][header.index(attribute)] for ind in list(perm)]
    inds = [ind for ind in range(len(inds)) if inds[ind] == clargs.subpop]
    inds = np.asarray(inds)
else:
    inds = []
    ins = [raw[ind][header.index(attribute)] for ind in list(perm)]
    inds.append(np.asarray(
        [ind for ind in range(len(ins)) if ins[ind] == clargs.subpop]))
    inds.append(np.asarray(
        [ind for ind in range(len(ins)) if ins[ind] == clargs.subpop2]))
    rs = [r[inds[0]], r[inds[1]]]
    ss = [s[inds[0]], s[inds[1]]]
    ss0 = [s0[inds[0]], s0[inds[1]]]

# Create directories as needed.
dir = 'unweighted'
if RANDOMIZE:
    dir += str(clargs.seed_for_rng)
try:
    os.mkdir(dir)
except FileExistsError:
    pass
dir += '/' + clargs.subpop
if clargs.subpop2 is not None:
    dir += '_' + clargs.subpop2
try:
    os.mkdir(dir)
except FileExistsError:
    pass
dir += '/'

# Plot reliability diagrams and the cumulative graph.
nin = [2, 5, 10, 20, 50]
for nbins in nin:
    # Construct a reliability diagram with bins whose widths are roughly equal.
    if clargs.subpop2 is None:
        filename = dir + 'equiscores' + str(nbins) + '.pdf'
        subpop_weighted.equiscores(r, s, inds, nbins, filename)
    else:
        filename = dir + 'equiscore' + str(nbins) + '.pdf'
        disjoint.equiscore(rs, ss, nbins, filename)
    # Construct a reliability diagram with bins whose error bars are similar.
    if clargs.subpop2 is None:
        filename = dir + 'equierrs' + str(nbins) + '.pdf'
        rng2 = default_rng(seed=987654321)
        nout = subpop_weighted.equierrs(r, s, inds, nbins, rng2, filename)
    else:
        filename = dir + 'equisamps' + str(nbins) + '.pdf'
        disjoint.equisamps(rs, ss, nbins, filename)
# Construct a plot of cumulative differences.
majorticks = 10
minorticks = 300
filename = dir + 'cumulative.pdf'
if clargs.subpop2 is None:
    kuiper, kolmogorov_smirnov, lenscale, ate = subpop_weighted.cumulative(
        r, s, inds, majorticks, minorticks, False, filename)
else:
    kuiper, kolmogorov_smirnov, lenscale, n = disjoint.cumulative(
        rs, ss, majorticks, minorticks, False, filename)
    ate = disjoint.ate(rs, ss, rng, num_rand=1)
    rate = disjoint.ate(rs, ss0, rng, num_rand=25)
# Calculate P-values.
kuiperp = 1 - dists.kuiper(kuiper / lenscale)
kolmogorov_smirnovp = 1 - dists.kolmogorov_smirnov(
    kolmogorov_smirnov / lenscale)
# Save metrics in a text file.
filename = dir + 'metrics.txt'
with open(filename, 'w') as f:
    if clargs.subpop2 is None:
        f.write('len(s) =\n')
        f.write(f'{len(s)}\n')
        if not RANDOMIZE:
            f.write('len(np.unique(s[inds])) =\n')
            f.write(f'{len(np.unique(s[inds]))}\n')
        f.write('len(inds) =\n')
        f.write(f'{len(inds)}\n')
    else:
        f.write('len(s0) =\n')
        f.write(f'{len(s0)}\n')
        f.write('len(np.unique(s0)) =\n')
        f.write(f'{len(np.unique(s0))}\n')
        f.write('len(ss[0]) =\n')
        f.write(f'{len(ss[0])}\n')
        f.write('len(ss[1]) =\n')
        f.write(f'{len(ss[1])}\n')
        f.write(f'n =\n')
        f.write(f'{n}\n')
        f.write('rate =\n')
        f.write(f'{rate:.4}\n')
        f.write('rate / lenscale =\n')
        f.write(f'{rate / lenscale:.4}\n')
    f.write('ate =\n')
    f.write(f'{ate:.4}\n')
    f.write('ate / lenscale =\n')
    f.write(f'{ate / lenscale:.4}\n')
    f.write('lenscale =\n')
    f.write(f'{lenscale:.4}\n')
    f.write('Kuiper =\n')
    f.write(f'{kuiper:.4}\n')
    f.write('Kolmogorov-Smirnov =\n')
    f.write(f'{kolmogorov_smirnov:.4}\n')
    f.write('Kuiper / lenscale =\n')
    f.write(f'{(kuiper / lenscale):.4}\n')
    f.write('Kolmogorov-Smirnov / lenscale =\n')
    f.write(f'{(kolmogorov_smirnov / lenscale):.4}\n')
    f.write('Kuiper P-value =\n')
    f.write(f'{kuiperp:.4}\n')
    f.write('Kolmogorov-Smirnov P-value =\n')
    f.write(f'{kolmogorov_smirnovp:.4}\n')
