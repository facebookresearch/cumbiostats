#!/usr/bin/env python3

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Performs a data analysis of the Behavioral Risk Factor Surveillance System

This script offers a command-line option, "--kitchen_sink", and the possibility
of specifying an integer number as the seed in the random number generator;
specifying "--kitchen_sink" causes the script to process 13 subpopulations
(rather than only the default 5). Both "--kitchen_sink" and specifying a number
in the command-line invocation are optional (the seed for the random number
defaults to 543216789).

This script plots reliability diagrams and cumulative differences between
subpopulations (where "subpopulation" can mean the full population, too). The
script also saves text files of numerical statistics associated with the plots.
The script creates directories, "weighted" and "weighted[random_seed]", within
the working directory if the directories do not already exist. In the first
directory, "weighted", the script creates subdirectories with names,
"[subpop]_and_[subpop]", corresponding to comparison of two subpopulations with
paired responses, if the subdirectories do not already exist. In each of these
subdirectories, "[subpop]_and_[subpop]", the script createst subdirectories,
"[covar]", for the covariates listed in covars defined below, with covars
defaulting to consisting of only the body mass index (BMI). In the second
directory, "weighted[random_seed]", the script creates subdirectories,
"[subpop]", corresponding to the subpopulations used. In each of these
subdirectories, "[subpop]", the script creates subdirectories,
"[response]_vs_[covar]", corresponding to comparison of two subpopulations,
along with comparison of each subpopulation to the full population, perturbing
at random the scores when comparing two subpopulations directly (to ensure that
the scores are distinct). In all cases, the scores are the covariate's values.
The response variates are in the lists revars and forpairs defined below, which
also define subpopulations considered. Different values for response variates
define the subpopulations for these subdirectories, "[response]_vs_[covar]", in
the subdirectories, "[subpop]", whereas the subdirectories mentioned earlier,
"[subpop]_and_[subpop]", compare different response variates for each
individual participant in the survey that generated the data set (each pair
of responses comes from a single individual). Again, creation of any of these
subdirectories is contingent on the subdirectories not already existing.

In the lowest-level subdirectories under the directory, "weighted", the script
creates reliability diagrams with the numbers of bins specified by the list
nbins defined below. Reliability diagrams whose bins are equispaced along the
scores (where the scores are the values of the covariate) get saved to files,
"equiscores[nbin].pdf", where "[nbin]" is the number of bins for each of the
subpopulations being compared, taking the values given by the list nbins.
Reliability diagrams whose bins have roughly equal-sized error bars get saved
to files, "equierrs[nbin].pdf", where again "[nbin]" gives the number of bins.
The script also saves the plot of cumulative differences to a file,
"cumulative.pdf", as well as a file, "metrics.txt", reporting numerical stats
associated with the plots and their associated data. The script overwrites any
existing files.

In the lowest-level directories under the directory, "weighted[random_seed]",
the script saves all the same files as under the directory, "weighted", but now
in three variants, "0", "1", and "01". The filenames with "0" appended compare
subpopulation 0 to the full population; the filenames with "1" appended compare
subpopulation 1 to the full population; and the filenames with "01" compare
the two subpopulations to each other directly. Thus, the naming convention for
the files becomes "equiscores01_[nbin].pdf", "equierrs01_[nbin].pdf", and
"cumulative01.pdf", and similarly for "0" and "1" rather than "01". There is
still only a single file, "metrics.txt", per lowest-level subdirectory.
Existing files (if any) get overwritten.

If the constant BMI is set to True (and the code below defaults to BMI = True),
so that the only covariate considered in the processing described above is BMI,
then the script creates a special extra subdirectory, "Computed_Sex_Variable",
of the directory, "weighted[random_seed]". Within this additional subdirectory,
the script creates a subdirectory whose naming convention involves centimeters,
"Computed_Body_Mass_Index_vs_Computed_Height_in_Centimeters", and saves files
analogous to those described in the previous paragraph, with the response now
being the BMI, with the covariate being the individual's height in centimeters,
and with the subpopulations under consideration being men and women. As before,
the script creates the directories only when they do not already exist, and
overwrites any existing files with the results of the latest processing.

The data analysis all concerns the Behavioral Risk Factor Surveillance Survey
of the Centers for Disease Control and Prevention. The data is available at
`data and docs <https://www.cdc.gov/brfss/data_documentation/index.htm>`_.
More specifically, the paper accompanying these codes uses the 2022 ASCII data,
`zipfile <https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022ASC.zip>`_.
All this data is in the public domain. If the decompressed data file,
"LLCP2022.txt", does not already exist in the working directory, this script
then starts by downloading the data and decompressing into the ASCII text file,
"LLCP2022.txt", removing the downloaded zip file after decompression.

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import argparse
import numpy as np
import os
import urllib.request
import zipfile
from numpy.random import default_rng

import disjoint_weighted
import dists
import paired_weighted
import subpop_weighted


# Set whether to use the body mass index as the only covariate.
BMI = True

# Set whether to print extra debugging information.
DEBUG = False

# Parse the command-line arguments (if any).
parser = argparse.ArgumentParser()
parser.add_argument('seed_for_rng', nargs='?', default=543216789, type=int)
parser.add_argument('--kitchen_sink', action='store_true')
clargs = parser.parse_args()

# Form the codebook for the BRFSS; the ordered pairs specify the indices
# of the columns which report the values of the corresponding variate.
codebook = {
    'State FIPS Code': (1, 2),
    'File Month': (17, 18),
    'Interview Month': (19, 20),
    'Interview Day': (21, 22),
    'Interview Year': (23, 26),
    'Final Disposition': (32, 35),
    'Annual Sequence Number': (36, 45),
    'Correct telephone number?': (63, 63),
    'Land Line Introduction': (64, 64),
    'Do you live in college housing?': (65, 65),
    'Resident of State': (66, 66),
    'Cellular Telephone': (67, 67),
    'Are you 18 years of age or older?': (68, 68),
    'Are you male or female?': (69, 69),
    'Number of Adults in Household': (70, 71),
    'Are you male or female?': (72, 72),
    'Number of Adult Men in Household': (73, 74),
    'Number of Adult Women in Household': (75, 76),
    'Respondent Selection': (77, 77),
    'Safe Time to Talk': (78, 78),
    'Correct phone number?': (79, 79),
    'Is this a cell phone?': (80, 80),
    'Are you 18 years of age or older?': (81, 81),
    'Are you male or female?': (82, 82),
    'Do you live in a private residence?': (83, 83),
    'Do you live in college housing?': (84, 84),
    'Do you currently live in _____(state)_____?': (85, 85),
    'Do you also have a landline telephone?': (88, 88),
    'Number of Adults in Household': (89, 90),
    'Sex of Respondent': (91, 91),
    'General Health': (101, 101),
    'Number of Days Physical Health Not Good': (102, 103),
    'Number of Days Mental Health Not Good': (104, 105),
    'Poor Physical or Mental Health': (106, 107),
    'What is primary source of health insurance?': (108, 109),
    'Have personal health care provider?': (110, 110),
    'Could Not Afford to See Doctor': (111, 111),
    'Length of Time Since Last Routine Checkup': (112, 112),
    'Exercise in Past 30 Days': (113, 113),
    'How Much Time Do You Sleep': (114, 115),
    'Last Visited Dentist or Dental Clinic': (116, 116),
    'Number of Permanent Teeth Removed': (117, 117),
    'Ever Diagnosed with Heart Attack': (118, 118),
    'Ever Diagnosed with Angina or Coronary Heart Disease': (119, 119),
    'Ever Diagnosed with a Stroke': (120, 120),
    'Ever Told Had Asthma': (121, 121),
    'Still Have Asthma': (122, 122),
    'Ever told you had skin cancer that is not melanoma?': (123, 123),
    'Ever told you had melanoma or any other types of cancer?': (124, 124),
    'Ever told you had COPD, emphysema, or chronic bronchitis?': (125, 125),
    'Ever told you had a depressive disorder': (126, 126),
    'Ever told you have kidney disease?': (127, 127),
    'Told Had Arthritis': (128, 128),
    'Ever Told You Had Diabetes': (129, 129),
    'Marital Status': (168, 168),
    'Education Level': (169, 169),
    'Own or Rent Home': (170, 170),
    'Household Landline Telephones': (179, 179),
    'Are You a Veteran': (182, 182),
    'Employment Status': (183, 183),
    'Number of Children in Household': (184, 185),
    'Income Level': (186, 187),
    'Pregnancy Status': (188, 188),
    'Reported Weight in Pounds': (189, 192),
    'Reported Height in Feet and Inches': (193, 196),
    'Are you deaf or do you have serious difficulty hearing?': (197, 197),
    'Blind or Difficulty Seeing': (198, 198),
    'Difficulty Concentrating or Remembering': (199, 199),
    'Difficulty Walking or Climbing Stairs': (200, 200),
    'Difficulty Dressing or Bathing': (201, 201),
    'Difficulty Doing Errands Alone': (202, 202),
    'Have You Ever Had a Mammogram': (203, 203),
    'How Long Since Last Mammogram': (204, 204),
    'Have you ever had a cervical cancer screening test?': (205, 205),
    'Time Since Last Cervical Cancer Screening Test': (206, 206),
    'Have a PAP Test and Recent Cervical Cancer Screening': (207, 207),
    'Have an H.P.V. Test and Recent Cervical Cancer Screening': (208, 208),
    'Had Hysterectomy': (209, 209),
    'Ever Had Sigmoidoscopy|Colonoscopy': (210, 210),
    'Ever Had a Colonoscopy, Sigmoidoscopy, or Both': (211, 211),
    'How Long Since You Had Colonoscopy': (212, 212),
    'How Long Since You Had Sigmoidoscopy': (213, 213),
    'Time Since Last Sigmoidoscopy|Colonoscopy': (214, 214),
    'Ever Had Any Other Kind of Test for Colorectal Cancer': (215, 215),
    'Ever Had a Virtual Colonoscopy': (216, 216),
    'How Long Since You Had Virtual Colonoscopy': (217, 217),
    'Ever Had Stool Test?': (218, 218),
    'How Long Since You Had Stool Test?': (219, 219),
    'Ever had stool DNA test?': (220, 220),
    'Was test part of Cologuard test?': (221, 221),
    'How Long Since You Had Stool DNA': (222, 222),
    'Smoked at Least 100 Cigarettes': (223, 223),
    'Frequency of Days Now Smoking': (224, 224),
    'Use of Smokeless Tobacco Products': (225, 225),
    'Do you now use e-cigarettes or vaping products every day?': (226, 226),
    'How old when you first started smoking?': (227, 229),
    'How old when you last smoked?': (230, 232),
    'On average, how many cigarettes do you smoke each day?': (233, 235),
    'Did you have a CT or CAT scan?': (236, 236),
    'Were any CT or CAT scans done to check for lung cancer?': (237, 237),
    'When did you have your most recent CT or CAT scan?': (238, 238),
    'Days in Past 30 Had Alcoholic Beverage': (239, 241),
    'Average Alcoholic Drinks Per Day in Past 30': (242, 243),
    'Binge Drinking': (244, 245),
    'Most Drinks on Single Occasion Past 30 Days': (246, 247),
    'Adult Flu Shot|Spray Past 12 Months': (248, 248),
    'When did you receive your most recent flu shot|spray?': (249, 254),
    'Pneumonia Shot Ever': (255, 255),
    'Received tetanus shot since 2005?': (256, 256),
    'Ever Tested H.I.V.': (257, 257),
    'Month and Year of Last HIV Test': (258, 263),
    'Do Any High Risk Situations Apply': (264, 264),
    'Have you ever been told you tested positive for COVID-19?': (265, 265),
    'Have a 3 month or longer COVID symptoms?': (266, 266),
    'Which was the primary symptom that you experienced?': (267, 268),
    'When was your last blood test for high blood sugar?': (269, 269),
    'Ever been told that you have borderline- or pre-diabetes?': (270, 270),
    'What type of diabetes do you have?': (271, 271),
    'Now Taking Insulin': (272, 272),
    'Times Checked for Glycosylated Hemoglobin': (273, 274),
    'Last Eye Exam Where Pupils Were Dilated': (275, 275),
    'When was the last time they took a photo of your retina?': (276, 276),
    'When was the last time you studied how to manage diabetes?': (277, 277),
    'Ever Had Feet Sores|Irritations Lasting More than 4 Weeks': (278, 278),
    'Told Had Chronic Fatigue Syndrome or ME': (279, 279),
    'Still Have Chronic Fatigue Syndrome or ME': (280, 280),
    'How Many Hours a Week Have You Been Able to Work': (281, 281),
    'Where did you get your last flu shot|vaccine?': (282, 283),
    'Have you ever had an H.P.V. vaccination?': (284, 284),
    'How many HPV shots did you receive?': (285, 286),
    'Have you ever had the shingles or zoster vaccine?': (287, 287),
    'Received At Least One COVID-19 Vaccination': (288, 288),
    'Will you get COVID-19 vaccination?': (289, 289),
    'Number of COVID-19 Vaccinations Received': (290, 290),
    'Intend to Get COVID-19 Vaccination': (291, 291),
    'Month|Year of First COVID-19 Vaccination': (292, 297),
    'Month|Year of Second COVID-19 Vaccination': (298, 303),
    'Did you have a cough?': (304, 304),
    'Did you cough up phlegm?': (305, 305),
    'Did you have shortness of breath?': (306, 306),
    'Have you ever been given a breathing test?': (307, 307),
    'How many years have you smoked tobacco products?': (308, 309),
    'How many types of cancer?': (310, 310),
    'Age Told Had Cancer': (311, 312),
    'Type of Cancer': (313, 314),
    'Currently Receiving Treatment for Cancer': (315, 315),
    'What Type of Doctor Provides Majority of Your Care': (316, 317),
    'Did You Receive a Summary of Cancer Treatments Received': (318, 318),
    'Ever Receive Instructions from a Dctor for Follow-Up': (319, 319),
    'Instructions Written or Printed': (320, 320),
    'Did Health Insurance Pay for All of Your Cancer Treatment': (321, 321),
    'Ever denied insurance coverage because of your cancer?': (322, 322),
    'Participate in clinical trial as part of cancer treatment?': (323, 323),
    'Currently have physical pain from cancer or treatment?': (324, 324),
    'Is pain under control?': (325, 325),
    'Ever Had PSA Test': (326, 326),
    'Time Since Most Recent PSA Test': (327, 327),
    'What was the main reason you had this PSA test?': (328, 328),
    'Who first suggested this PSA test?': (329, 329),
    'Did You Talk About Advantages or Disadvantages of PSA Test': (330, 330),
    'Have you experienced confusion|memory-loss getting worse?': (331, 331),
    'Given Up Day-to-Day Chores Due to Confusion or Memory Loss': (332, 332),
    'Need Assistance with Day-to-Day Activity Due to Confusion': (333, 333),
    'When You Need Help with Day-to-Day Activity You Can Get It': (334, 334),
    'Does Memory Loss Interfere with Work or Social Activities': (335, 335),
    'Have you discussed your confusion with a health-care pro?': (336, 336),
    'Provided Regular Care for Family or Friend': (337, 337),
    'Relationship of person to whom you are giving care?': (338, 339),
    'How Long Provided Care for Person': (340, 340),
    'How many hours do you provide care for person?': (341, 341),
    'What is the major health problem for care for person?': (342, 343),
    'Does person being cared for have Alzheimer\'s disease?': (344, 344),
    'Managed Personal Care': (345, 345),
    'Managed Household Tasks': (346, 346),
    'Do you expect to have a relative you provide care for?': (347, 347),
    'Live with anyone depressed, mentally ill, or suicidal?': (348, 348),
    'Live with a problem drinker|alcoholic?': (349, 349),
    'Live with anyone who used drugs or abused prescriptions?': (350, 350),
    'Live with anyone who served time in prison or jail?': (351, 351),
    'Were your parents divorced|separated?': (352, 352),
    'How often did your parents beat up each other?': (353, 353),
    'How often did a parent physically hurt you in any way?': (354, 354),
    'How often did a parent swear at you?': (355, 355),
    'How often did anyone ever touch you sexually?': (356, 356),
    'How often did anyone make you touch them sexually?': (357, 357),
    'How often did anyone ever force you to have sex?': (358, 358),
    'Did an Adult Make You Feel Safe and Protected': (359, 359),
    'Did an Adult Make Sure Basic Needs Were Met': (360, 360),
    'Satisfaction with Life': (361, 361),
    'How Often Get Emotional Support Needed': (362, 362),
    'How often do you feel socially isolated from others?': (363, 363),
    'Have you lost employment or had hours reduced?': (364, 364),
    'During the Past 12 Months Have You Received Food Stamps': (365, 365),
    'How often did the food you bought not last?': (366, 366),
    'Were you not able to pay your bills?': (367, 367),
    'Were you not able to pay utilities or risk losing service?': (368, 368),
    'Have you lacked reliable transportation you need?': (369, 369),
    'How often have you felt this kind of stress?': (370, 370),
    'During the past 30 days, on how many days did you use pot?': (371, 372),
    'Did you smoke marijuana or cannabis?': (373, 373),
    'Did you eat marijuana or cannabis?': (374, 374),
    'Did you vape marijuana or cannabis?': (375, 375),
    'Did you dab marijuana or cannabis?': (376, 376),
    'Did you ever use marijuana or cannabis some other way?': (377, 377),
    'Interval Since Last Smoked': (379, 380),
    'Stopped Smoking in Past 12 Months': (381, 381),
    'Do you usually smoke menthol cigarettes?': (382, 382),
    'Do you usually use menthol e-cigarettes?': (383, 383),
    'Have you heard of heated tobacco products?': (384, 384),
    'Asked During Checkup If You Drink Alcohol': (385, 385),
    'Asked in person or by form how much you drink?': (386, 386),
    'Asked if you drank 4+(women) or 5+(men) alcoholic drinks?': (387, 387),
    'Offered advice about what level of drinking is risky?': (388, 388),
    'Were you advised to reduce or quit your drinking?': (389, 389),
    'Any Firearms in Home': (390, 390),
    'Any Firearms Loaded': (391, 391),
    'Any loaded firearms also unlocked?': (392, 392),
    'Gender of Child': (599, 599),
    'Child\'s Sex at Birth': (600, 600),
    'Relationship to Child': (635, 635),
    'Health Professional Ever Said Child Has Asthma': (636, 636),
    'Child still have asthma?': (637, 637),
    'Are you male or female?': (638, 638),
    'Sexual Orientation for Males': (639, 639),
    'Sexual Orientation for Females': (640, 640),
    'Do you consider yourself to be transgender?': (641, 641),
    'Have you had sexual intercourse in the past 12 months?': (642, 642),
    'Did you do anything to keep from getting pregnant?': (643, 643),
    'What did you do to keep you from getting pregnant?': (644, 645),
    'Are you doing anything to keep from getting pregnant?': (646, 647),
    'Where did you get what you used to prevent pregnancy?': (648, 649),
    'Why did you not do anything to keep from getting pregnant?': (650, 651),
    'What is your preferred birth control method?': (652, 653),
    'How do other people usually classify you in this country?': (654, 655),
    'How often do you think about your race?': (656, 656),
    'Were you treated better|worse than people of other races?': (657, 657),
    'How were treated at work compared to those of other races?': (658, 658),
    'Did your race help|hurt the process of seeking healthcare?': (659, 659),
    'Times in Past Month Racist Treatment Manifested Physically': (660, 660),
    'Questionnaire Version Identifier': (665, 666),
    'Language Identifier': (667, 668),
    'Metropolitan Status': (1402, 1402),
    'Urban|Rural Status': (1403, 1403),
    'Metropolitan Status Code': (1409, 1409),
    'Sample Design Stratification Variable': (1410, 1415),
    'Stratum Weight': (1416, 1425),
    'Raw Weighting Factor Used in Raking': (1446, 1455),
    'Design Weight Use in Raking': (1456, 1465),
    'Imputed Race|Ethnicity Value': (1471, 1472),
    'Child Hispanic, Latino|a, or Spanish Origin Calculated': (1482, 1482),
    'Child Non-Hispanic Race Including Multiracial': (1539, 1540),
    'Preferred Child Race Categories': (1541, 1542),
    'Four Level Child Age': (1569, 1569),
    'Final Child Weight: Land-line and Cell-phone Data': (1585, 1594),
    'Dual Phone Use Categories': (1682, 1682),
    'Dual Phone Use Correction Factor': (1683, 1692),
    'Truncated Design Weight Used in Combined Raking': (1693, 1702),
    'Final Weight: Land-line and Cell-phone Data': (1751, 1760),
    'Adults with Good or Better Health': (1899, 1899),
    'Computed Physical Health Status': (1900, 1900),
    'Computed Mental Health Status': (1901, 1901),
    'Have Any Health Insurance': (1902, 1902),
    'Respondents Aged 18-64 with Health Insurance': (1903, 1903),
    'Leisure Time Physical Activity Calculated Variable': (1904, 1904),
    'Adults Aged 18+ That Have Had Permanent Teeth Extracted': (1905, 1905),
    'Adults Aged 65+ Who Have Had All Their Teeth Extracted': (1906, 1906),
    'Adults Who Have Visited a Dentist within the Past Year': (1907, 1907),
    'Ever Had Coronary Heart Disease | Myocardial Infarcation': (1908, 1908),
    'Lifetime Asthma Calculated Variable': (1909, 1909),
    'Current Asthma Calculated Variable': (1910, 1910),
    'Computed Asthma Status': (1911, 1911),
    'Respondents Diagnosed with Arthritis': (1912, 1912),
    'Computed Preferred Race': (1969, 1970),
    'Calculated Non-Hispanic Race Including Multiracial': (1971, 1972),
    'Hispanic, Latino|a, or Spanish Origin Calculated': (1975, 1975),
    'Computed Race-Ethnicity Grouping': (1976, 1976),
    'White Non-Hispanic Race Group': (1977, 1977),
    'Computed Five Level Race|Ethnicity Category': (1978, 1978),
    'Computed Race Groups Used for Internet Prevalence Tables': (1979, 1979),
    'Computed Sex Variable': (1980, 1980),
    'Reported Age in Five-Year Age Categories Calculated': (1981, 1982),
    'Reported Age in Two Age Groups Calculated': (1983, 1983),
    'Imputed Age Value Collapsed Above 80': (1984, 1985),
    'Imputed Age in Six Groups': (1986, 1986),
    'Computed Height in Inches': (1987, 1989),
    'Computed Height in Meters': (1990, 1992),
    'Computed Weight in Kilograms': (1993, 1997),
    'Computed Body Mass Index': (1998, 2001),
    'Computed Body Mass Index Categories': (2002, 2002),
    'Overweight or Obese Calculated Variable': (2003, 2003),
    'Computed Number of Children in Household': (2004, 2004),
    'Computed Level of Education Completed Categories': (2005, 2005),
    'Computed Income Categories': (2006, 2006),
    'Women Respondents Aged 40+ Who Have Had a Mammogram': (2007, 2007),
    'Women Respondents Aged 50-74 Who Have Had a Mammogram': (2008, 2008),
    'Had Colonoscopy Calculated Variable': (2009, 2009),
    'Respondents Aged 45-75 Who Have Had a Colonoscopy': (2010, 2010),
    'Had Sigmoidoscopy Calculated Variable': (2011, 2011),
    'Respondents Aged 45-75 Who Had a 5-Year Sigmoidoscopy': (2012, 2012),
    'Respondents Aged 45-75 Who Had a 10-Year Sigmoidoscopy': (2013, 2013),
    'Respondents Aged 45-75 Who Have Had a Stool Test': (2014, 2014),
    'Respondents Aged 45-75 Who Have Had a Stool DNA Test': (2015, 2015),
    'Respondents Aged 45-75 Who Had a Virtual Colonoscopy': (2016, 2016),
    'Respondents Aged 45-75 Who Had Sigmoidoscopy+Stool-Test': (2017, 2017),
    'Respondents Aged 45-75 Who Met All USPSTF Recommendation': (2018, 2018),
    'Computed Smoking Status': (2019, 2019),
    'Current Smoking Calculated Variable': (2020, 2020),
    'Current E-Cigarette User Calculated Variable': (2021, 2021),
    'Number of Years Smoked Cigarettes': (2022, 2024),
    'Number of Packs of Cigarettes Smoked Per Day': (2025, 2029),
    'Years Smoked Reported Packs Per Day': (2030, 2033),
    'Number of Years Since Quit Smoking Cigarettes': (2034, 2035),
    'Smoking Group': (2036, 2036),
    'Lung Cancer Screening Recommendation Status': (2037, 2037),
    'Drink Any Alcoholic Beverages in Past 30 Days': (2038, 2038),
    'Computed Drink-Occasions-Per-Day': (2039, 2041),
    'Binge Drinking Calculated Variable': (2042, 2042),
    'Computed Number of Alcoholic Beverages Per Week': (2043, 2047),
    'Heavy Alcohol Consumption Calculated Variable': (2048, 2048),
    'Flu Shot Calculated Variable': (2049, 2049),
    'Pneumonia Vaccination Calculated Variable': (2050, 2050),
    'Ever Been Tested for HIV Calculated Variable': (2051, 2051),
}

# Check that the codebook has no obvious errors.
diff = []
size = []
prev = 0
for val in sorted(codebook.values()):
    diff.append(val[0] - prev)
    prev = val[1]
    size.append(prev - val[0])
for entry in diff:
    assert(entry >= 1)
assert sum(diff) + sum(size) == 2051

# Set the filename containing the data from the BRFSS of the CDC.
filename = 'LLCP2022.txt'

# Download the data if necessary.
if not os.path.isfile(filename):
    print('downloading data...')
    filezip = 'LLCP2022ASC.zip'
    urlzip = 'https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022ASC.zip'
    urllib.request.urlretrieve(urlzip, filezip)
    print('decompressing data...')
    original = 'LLCP2022.ASC '
    with zipfile.ZipFile(filezip, 'r') as zip_ref:
        zip_ref.extract(original)
    os.remove(filezip)
    os.rename(original, filename)

# Determine the number of rows in the file for filename.
lines = 0
with open(filename, 'r') as f:
    for line in f:
        lines += 1
print(f'lines = {lines}')

# Read the data.
col = {}
data = np.zeros((lines, len(codebook)))
with open(filename, 'r') as f:
    for line_num, line in enumerate(f):
        for column, (key, val) in enumerate(codebook.items()):
            datum = line[(val[0] - 1):val[1]].strip(' ')
            data[line_num, column] = float(datum) if datum != '' else -1
            col[key] = column

# Verify that all weights are strictly positive.
assert np.all(data[:, col['Final Weight: Land-line and Cell-phone Data']] > 0)

print('data[:, col[\'Final Weight: Land-line and Cell-phone Data\']] =')
print(data[:, col['Final Weight: Land-line and Cell-phone Data']])

print('len(np.unique(Computed Weight in Kilograms)) =')
print(len(np.unique(data[:, col['Computed Weight in Kilograms']])))

print('len(np.unique(Computed Body Mass Index)) =')
print(len(np.unique(data[:, col['Computed Body Mass Index']])))

print('np.unique(Computed Weight in Pounds) =')
print(np.unique(data[:, col['Reported Weight in Pounds']]))

print('np.unique(Computed Body Mass Index) =')
print(np.unique(data[:, col['Computed Body Mass Index']]))

print('np.histogram(Computed Weight in Kilograms) =')
print(np.histogram(
    data[:, col['Computed Weight in Kilograms']],
    bins=np.unique(data[:, col['Computed Weight in Kilograms']])[::20]))

print('np.histogram(Computed Body Mass Index) =')
print(np.histogram(
    data[:, col['Computed Body Mass Index']],
    bins=np.unique(data[:, col['Computed Body Mass Index']])[::120]))

# Tabulate all covariates of interest together with factors for normalization.
if BMI:
    covars = {'Computed Body Mass Index': .01}
else:
    covars = {
        'Computed Body Mass Index': .01,
        'Computed Weight in Kilograms': .01,
        'Computed Height in Meters': .01,
    }

# Filter out individuals for whom any of the covariates is non-positive.
for covar in covars:
    data = data[data[:, col[covar]] > 0, :]
print(f'data.shape = {data.shape}')

# Read out the weights.
w0 = data[:, col['Final Weight: Land-line and Cell-phone Data']]

# Tabulate all subpopulations of interest, as the associated response variate
# together with a pair of coded values of the response variate that specifies
# two subpopulations.
if clargs.kitchen_sink:
    subpops = {
        'Ever Diagnosed with Heart Attack': (1, 2),
        'Ever Diagnosed with Angina or Coronary Heart Disease': (1, 2),
        'Ever Diagnosed with a Stroke': (1, 2),
        'Ever Told Had Asthma': (1, 2),
        'Ever told you have kidney disease?': (1, 2),
        'Have you ever been told you tested positive for COVID-19?': (1, 2),
        'Have a 3 month or longer COVID symptoms?': (1, 2),
        'Received At Least One COVID-19 Vaccination': (1, 2),
        'Could Not Afford to See Doctor': (1, 2),
        'During the Past 12 Months Have You Received Food Stamps': (1, 2),
        'What type of diabetes do you have?': (1, 2),
        'Metropolitan Status': (1, 2),
        'Ever Been Tested for HIV Calculated Variable': (1, 2),
    }
else:
    subpops = {
        'Ever Diagnosed with Heart Attack': (1, 2),
        'Ever Diagnosed with a Stroke': (1, 2),
        'Ever told you have kidney disease?': (1, 2),
        'Could Not Afford to See Doctor': (1, 2),
        'Ever Been Tested for HIV Calculated Variable': (1, 2),
    }

# Tabulate all response variates of interest.
revars = list(subpops.keys())

# Tabulate all pairs of response variates of interest for paired comparisons.
forpairs = [
    'Ever Diagnosed with Heart Attack',
    'Ever Diagnosed with Angina or Coronary Heart Disease',
    'Ever Diagnosed with a Stroke',
    'Ever told you have kidney disease?',
]
pairs = [
    (first, second) for first in forpairs
    for second in [item for item in forpairs if item != first]]


# Create a directory as needed.
dir = 'weighted'
try:
    os.mkdir(dir)
except FileExistsError:
    pass
dir += '/'

# Process all paired samples.
for pair in pairs:
    # Create a subdirectory as needed.
    subvardir = dir + pair[0] + '_and_' + pair[1]
    subvardir = subvardir.replace(' ', '_')
    subvardir = subvardir.replace('?', '')
    try:
        os.mkdir(subvardir)
    except FileExistsError:
        pass
    subvardir += '/'

    # Process all covariates.
    for covar in covars:
        # Create a (sub)subdirectory as needed.
        subdir = subvardir + covar
        subdir = subdir.replace(' ', '_')
        subdir = subdir.replace('?', '')
        print('generating ' + subdir + '...')
        try:
            os.mkdir(subdir)
        except FileExistsError:
            pass
        subdir += '/'
        # Collect the scores.
        s = data[:, col[covar]] * covars[covar]
        # Collect the responses.
        r0 = 1. * (data[:, col[pair[0]]] == 1)
        r1 = 1. * (data[:, col[pair[1]]] == 1)
        # Sort the scores.
        perm = np.argsort(s, kind='stable')
        s = s[perm]
        r0 = r0[perm]
        r1 = r1[perm]
        w = w0[perm]
        # Set plotting parameters.
        majorticks = 8
        minorticks = 100
        nbins = [2, 5, 10, 20, 40]
        # Construct plots comparing the pairs of responses.
        filename = subdir + 'cumulative.pdf'
        kuiper, kolmogorov_smirnov, lenscale = paired_weighted.cumulative(
            r0, r1, s, majorticks, minorticks, filename, weights=w)
        kuiperss = kuiper / lenscale
        kolmogorov_smirnovss = kolmogorov_smirnov / lenscale
        kuipersp = 1 - dists.kuiper(kuiperss)
        kolmogorov_smirnovsp = dists.kolmogorov_smirnov(kolmogorov_smirnovss)
        kolmogorov_smirnovsp = 1 - kolmogorov_smirnovsp
        # Construct the reliability diagrams.
        for nbin in nbins:
            filename = subdir + 'equiscores' + str(nbin) + '.pdf'
            paired_weighted.equiscores(r0, r1, s, nbin, filename, weights=w)
            filename = subdir + 'equierrs' + str(nbin) + '.pdf'
            rng2 = default_rng(seed=987654321)
            paired_weighted.equierrs(
                r0, r1, s, nbin, rng2, filename, weights=w)
        # Evaluate the average treatment effect.
        mean0 = np.sum(r0 * w / w.sum())
        mean1 = np.sum(r1 * w / w.sum())
        ate = mean0 - mean1
        # Report metrics in a text file.
        filename = subdir + 'metrics.txt'
        with open(filename, 'w') as f:
            f.write('len(s) =\n')
            f.write(f'{len(s)}\n')
            f.write('len(np.unique(s)) =\n')
            f.write(f'{len(np.unique(s))}\n')
            f.write('ate =\n')
            f.write(f'{ate:.4}\n')
            f.write('ate / lenscale =\n')
            f.write(f'{ate / lenscale:.4}\n')
            f.write('lenscale =\n')
            f.write(f'{lenscale:.4}\n')
            f.write('kuiper =\n')
            f.write(f'{kuiper:.4}\n')
            f.write('kolmogorov_smirnov =\n')
            f.write(f'{kolmogorov_smirnov:.4}\n')
            f.write('kuiperss =\n')
            f.write(f'{kuiperss:.4}\n')
            f.write('kolmogorov_smirnovss =\n')
            f.write(f'{kolmogorov_smirnovss:.4}\n')
            f.write('kuipersp =\n')
            f.write(f'{kuipersp:.4}\n')
            f.write('kolmogorov_smirnovsp =\n')
            f.write(f'{kolmogorov_smirnovsp:.4}\n')


# Create a directory as needed.
dir = 'weighted' + str(clargs.seed_for_rng)
try:
    os.mkdir(dir)
except FileExistsError:
    pass
dir += '/'

# Process all subpopulations.
for subvar, vals in subpops.items():
    # Select the subpopulations.
    indices = tuple(np.nonzero(data[:, col[subvar]] == val)[0] for val in vals)

    # Create a subdirectory as needed.
    subvardir = dir + subvar
    subvardir = subvardir.replace(' ', '_')
    subvardir = subvardir.replace('?', '')
    try:
        os.mkdir(subvardir)
    except FileExistsError:
        pass
    subvardir += '/'

    # Process all relevant pairs of covariates and response variates.
    for covar in covars:
        revarss = revars.copy()
        revarss.remove(subvar)
        for revar in revarss:
            # Create a (sub)subdirectory as needed.
            subdir = subvardir + revar + '_vs_' + covar
            subdir = subdir.replace(' ', '_')
            subdir = subdir.replace('?', '')
            print('generating ' + subdir + '...')
            try:
                os.mkdir(subdir)
            except FileExistsError:
                pass
            subdir += '/'
            # Initialize the random number generator.
            rng = default_rng(seed=clargs.seed_for_rng)
            # Collect the scores.
            s = data[:, col[covar]] * covars[covar]
            s0 = np.copy(s)
            s += (2 * rng.permutation(s.size) / s.size - 1) * 1e-8
            if DEBUG:
                u, counts = np.unique(s, return_counts=True)
                duplicates = u[counts > 1]
                print(f'covar = ' + covar)
                print(f'revar = ' + revar)
                print(f'duplicates = {duplicates}')
                print(f'len(np.unique(s)) = {len(np.unique(s))}')
                print(f'len(s) = {len(s)}')
                print(f's = {s}')
                print(f's0 = {s0}')
            assert np.unique(s).size == s.size
            assert np.allclose(s, s0, atol=1e-6)
            # Collect the responses.
            r = 1. * (data[:, col[revar]] == 1)
            r0 = np.copy(r)
            # Sort the scores.
            perm = np.argsort(s, kind='stable')
            perm0 = np.argsort(s0, kind='stable')
            s = s[perm]
            s00 = s0[perm0]
            s0 = s0[perm]
            r = r[perm]
            r00 = r0[perm0]
            w = w0[perm]
            w00 = w0[perm0]
            # Construct the inverse permutations.
            permi = np.copy(perm)
            for ind in range(perm.size):
                permi[perm[ind]] = ind
            permi0 = np.copy(perm0)
            for ind in range(perm0.size):
                permi0[perm0[ind]] = ind
            # Determine the indices of the subpopulations.
            inds = tuple(np.sort(permi[ind]) for ind in indices)
            inds00 = tuple(np.sort(permi0[ind]) for ind in indices)
            # Set plotting parameters.
            majorticks = 8
            minorticks = 100
            nbins = [2, 5, 10, 20, 40]
            # Construct plots comparing the subpopulations to the full pop.
            kuipers = []
            kolmogorov_smirnovs = []
            kuiperss = []
            kolmogorov_smirnovss = []
            kuipersp = []
            kolmogorov_smirnovsp = []
            lenscales = []
            ates = []
            for subpop in range(len(inds)):
                # Construct the cumulative plot.
                filename = subdir + 'cumulative' + str(subpop) + '.pdf'
                kuiper, kolmogorov_smirnov, lenscale, wate = \
                    subpop_weighted.cumulative(
                        r00, s00, inds00[subpop], majorticks, minorticks, True,
                        filename, weights=w00)
                kuipers.append(kuiper)
                kolmogorov_smirnovs.append(kolmogorov_smirnov)
                kuiperss.append(kuiper / lenscale)
                kolmogorov_smirnovss.append(kolmogorov_smirnov / lenscale)
                kuipersp.append(1 - dists.kuiper(kuiper / lenscale))
                kolmogorov_smirnovsp.append(1 - dists.kolmogorov_smirnov(
                    kolmogorov_smirnov / lenscale))
                lenscales.append(lenscale)
                ates.append(wate)
                # Construct the reliability diagrams.
                for nbin in nbins:
                    filename = subdir + 'equiscores' + str(subpop)
                    filename += '_' + str(nbin) + '.pdf'
                    subpop_weighted.equiscores(
                        r00, s00, inds00[subpop], nbin, filename, weights=w00,
                        left=np.min(s00), right=np.max(s00), top=1)
                    filename = subdir + 'equierrs' + str(subpop)
                    filename += '_' + str(nbin) + '.pdf'
                    rng2 = default_rng(seed=987654321)
                    subpop_weighted.equierrs(
                        r00, s00, inds00[subpop], nbin, rng2, filename,
                        weights=w00)
            # Construct plots comparing the two subpopulations directly.
            rs = [r[inds[0]], r[inds[1]]]
            ss = [s[inds[0]], s[inds[1]]]
            ss0 = [s0[inds[0]], s0[inds[1]]]
            ws = [w[inds[0]], w[inds[1]]]
            filename = subdir + 'cumulative01.pdf'
            kuiper, kolmogorov_smirnov, lenscale, m = \
                disjoint_weighted.cumulative(
                    rs, ss, majorticks, minorticks, False, filename,
                    weights=ws)
            kuipers.append(kuiper)
            kolmogorov_smirnovs.append(kolmogorov_smirnov)
            kuiperss.append(kuiper / lenscale)
            kolmogorov_smirnovss.append(kolmogorov_smirnov / lenscale)
            kuipersp.append(1 - dists.kuiper(kuiper / lenscale))
            kolmogorov_smirnovsp.append(1 - dists.kolmogorov_smirnov(
                kolmogorov_smirnov / lenscale))
            lenscales.append(lenscale)
            # Construct the reliability diagrams.
            for nbin in nbins:
                filename = subdir + 'equiscores01' + '_' + str(nbin) + '.pdf'
                disjoint_weighted.equiscores(
                    rs, ss, nbin, filename, weights=ws)
                filename = subdir + 'equierrs01' + '_' + str(nbin) + '.pdf'
                rng2 = default_rng(seed=987654321)
                disjoint_weighted.equierrs(
                    rs, ss, nbin, rng2, filename, weights=ws)
            # Evaluate the average treatment effect, noting that the scores ss
            # are unique, so no further randomization is necessary.
            ates.append(disjoint_weighted.ate(
                rs, ss, rng, weights=ws, num_rand=1))
            ates.append(disjoint_weighted.ate(
                rs, ss0, rng, weights=ws, num_rand=25))
            # Report metrics in a text file.
            filename = subdir + 'metrics.txt'
            with open(filename, 'w') as f:
                f.write('len(r) =\n')
                f.write(f'{len(r)}\n')
                f.write('len(s) =\n')
                f.write(f'{len(s)}\n')
                f.write('len(np.unique(s0)) =\n')
                f.write(f'{len(np.unique(s0))}\n')
                f.write('tuple(len(ind) for ind in inds) =\n')
                f.write(f'{tuple(len(ind) for ind in inds)}\n')
                f.write('m =\n')
                f.write(f'{m}\n')
                f.write('ates =\n')
                f.write(f'({ates[0]:.4}, ')
                f.write(f'{ates[1]:.4}, ')
                f.write(f'{ates[2]:.4}, ')
                f.write(f'{ates[3]:.4})\n')
                f.write('ates / lenscales =\n')
                f.write(f'({ates[0] / lenscales[0]:.4}, ')
                f.write(f'{ates[1] / lenscales[1]:.4}, ')
                f.write(f'{ates[2] / lenscales[2]:.4}, ')
                f.write(f'{ates[3] / lenscales[2]:.4})\n')
                f.write('lenscales =\n')
                f.write(f'({lenscales[0]:.4}, ')
                f.write(f'{lenscales[1]:.4}, ')
                f.write(f'{lenscales[2]:.4})\n')
                f.write('kuipers =\n')
                f.write(f'({kuipers[0]:.4}, ')
                f.write(f'{kuipers[1]:.4}, ')
                f.write(f'{kuipers[2]:.4})\n')
                f.write('kolmogorov_smirnovs =\n')
                f.write(f'({kolmogorov_smirnovs[0]:.4}, ')
                f.write(f'{kolmogorov_smirnovs[1]:.4}, ')
                f.write(f'{kolmogorov_smirnovs[2]:.4})\n')
                f.write('kuiperss =\n')
                f.write(f'({kuiperss[0]:.4}, ')
                f.write(f'{kuiperss[1]:.4}, ')
                f.write(f'{kuiperss[2]:.4})\n')
                f.write('kolmogorov_smirnovss =\n')
                f.write(f'({kolmogorov_smirnovss[0]:.4}, ')
                f.write(f'{kolmogorov_smirnovss[1]:.4}, ')
                f.write(f'{kolmogorov_smirnovss[2]:.4})\n')
                f.write('kuipersp =\n')
                f.write(f'({kuipersp[0]:.4}, ')
                f.write(f'{kuipersp[1]:.4}, ')
                f.write(f'{kuipersp[2]:.4})\n')
                f.write('kolmogorov_smirnovsp =\n')
                f.write(f'({kolmogorov_smirnovsp[0]:.4}, ')
                f.write(f'{kolmogorov_smirnovsp[1]:.4}, ')
                f.write(f'{kolmogorov_smirnovsp[2]:.4})\n')


# Analyze BMI versus height in centimeters for men versus women.
if BMI:
    # Select the subpopulations.
    subvar = 'Computed Sex Variable'
    vals = (1, 2)
    indices = tuple(np.nonzero(data[:, col[subvar]] == val)[0] for val in vals)

    # Create a subdirectory as needed.
    subvardir = dir + subvar
    subvardir = subvardir.replace(' ', '_')
    subvardir = subvardir.replace('?', '')
    try:
        os.mkdir(subvardir)
    except FileExistsError:
        pass
    subvardir += '/'

    # Set the covariates and responses to consider.
    covar = 'Computed Height in Meters'
    covarlabel = 'Computed Height in Centimeters'
    revar = 'Computed Body Mass Index'
    revarscale = .01
    # Create a (sub)subdirectory as needed.
    subdir = subvardir + revar + '_vs_' + covarlabel
    subdir = subdir.replace(' ', '_')
    subdir = subdir.replace('?', '')
    print('generating ' + subdir + '...')
    try:
        os.mkdir(subdir)
    except FileExistsError:
        pass
    subdir += '/'
    # Initialize the random number generator.
    rng = default_rng(seed=clargs.seed_for_rng)
    # Collect the scores.
    s = data[:, col[covar]]
    s0 = np.copy(s)
    # Perturb the scores at random.
    s += (2 * rng.permutation(s.size) / s.size - 1) * 1e-6
    if DEBUG:
        u, counts = np.unique(s, return_counts=True)
        duplicates = u[counts > 1]
        print(f'covar = ' + covar)
        print(f'revar = ' + revar)
        print(f'duplicates = {duplicates}')
        print(f'len(np.unique(s)) = {len(np.unique(s))}')
        print(f'len(s) = {len(s)}')
        print(f's = {s}')
        print(f's0 = {s0}')
    assert np.unique(s).size == s.size
    assert np.allclose(s, s0, atol=1e-4)
    # Collect the responses.
    r = data[:, col[revar]] * revarscale
    r0 = np.copy(r)
    # Sort the scores.
    perm = np.argsort(s, kind='stable')
    perm0 = np.argsort(s0, kind='stable')
    s = s[perm]
    s00 = s0[perm0]
    s0 = s0[perm]
    r = r[perm]
    r00 = r0[perm0]
    w = w0[perm]
    w00 = w0[perm0]
    # Construct the inverse permutations.
    permi = np.copy(perm)
    for ind in range(perm.size):
        permi[perm[ind]] = ind
    permi0 = np.copy(perm0)
    for ind in range(perm0.size):
        permi0[perm0[ind]] = ind
    # Determine the indices of the subpopulations.
    inds = tuple(np.sort(permi[ind]) for ind in indices)
    inds00 = tuple(np.sort(permi0[ind]) for ind in indices)
    # Set plotting parameters.
    majorticks = 8
    minorticks = 100
    nbins = [2, 5, 10, 20, 40]
    # Construct plots comparing the subpopulations to the full pop.
    kuipers = []
    kolmogorov_smirnovs = []
    kuiperss = []
    kolmogorov_smirnovss = []
    kuipersp = []
    kolmogorov_smirnovsp = []
    lenscales = []
    ates = []
    for subpop in range(len(inds)):
        # Construct the cumulative plot.
        filename = subdir + 'cumulative' + str(subpop) + '.pdf'
        kuiper, kolmogorov_smirnov, lenscale, wate = \
            subpop_weighted.cumulative(
                r00, s00, inds00[subpop], majorticks, minorticks, False,
                filename, weights=w00)
        kuipers.append(kuiper)
        kolmogorov_smirnovs.append(kolmogorov_smirnov)
        kuiperss.append(kuiper / lenscale)
        kolmogorov_smirnovss.append(kolmogorov_smirnov / lenscale)
        kuipersp.append(1 - dists.kuiper(kuiper / lenscale))
        kolmogorov_smirnovsp.append(1 - dists.kolmogorov_smirnov(
            kolmogorov_smirnov / lenscale))
        lenscales.append(lenscale)
        ates.append(wate)
        # Construct the reliability diagrams.
        for nbin in nbins:
            filename = subdir + 'equiscores' + str(subpop)
            filename += '_' + str(nbin) + '.pdf'
            subpop_weighted.equiscores(
                r00, s00, inds00[subpop], nbin, filename, weights=w00,
                left=np.min(s00), right=np.max(s00), top=1)
            filename = subdir + 'equierrs' + str(subpop)
            filename += '_' + str(nbin) + '.pdf'
            rng2 = default_rng(seed=987654321)
            subpop_weighted.equierrs(
                r00, s00, inds00[subpop], nbin, rng2, filename, weights=w00)
    # Construct plots comparing the two subpopulations directly.
    rs = [r[inds[0]], r[inds[1]]]
    ss = [s[inds[0]], s[inds[1]]]
    ss0 = [s0[inds[0]], s0[inds[1]]]
    ws = [w[inds[0]], w[inds[1]]]
    filename = subdir + 'cumulative01.pdf'
    kuiper, kolmogorov_smirnov, lenscale, m = disjoint_weighted.cumulative(
        rs, ss, majorticks, minorticks, False, filename, weights=ws)
    kuipers.append(kuiper)
    kolmogorov_smirnovs.append(kolmogorov_smirnov)
    kuiperss.append(kuiper / lenscale)
    kolmogorov_smirnovss.append(kolmogorov_smirnov / lenscale)
    kuipersp.append(1 - dists.kuiper(kuiper / lenscale))
    kolmogorov_smirnovsp.append(1 - dists.kolmogorov_smirnov(
        kolmogorov_smirnov / lenscale))
    lenscales.append(lenscale)
    # Construct the reliability diagrams.
    for nbin in nbins:
        filename = subdir + 'equiscores01' + '_' + str(nbin) + '.pdf'
        disjoint_weighted.equiscores(rs, ss, nbin, filename, weights=ws)
        filename = subdir + 'equierrs01' + '_' + str(nbin) + '.pdf'
        rng2 = default_rng(seed=987654321)
        disjoint_weighted.equierrs(rs, ss, nbin, rng2, filename, weights=ws)
    # Evaluate the average treatment effect, noting that the scores ss
    # are unique, so no further randomization is necessary.
    ates.append(disjoint_weighted.ate(rs, ss, rng, weights=ws, num_rand=1))
    ates.append(disjoint_weighted.ate(rs, ss0, rng, weights=ws, num_rand=25))
    # Report metrics in a text file.
    filename = subdir + 'metrics.txt'
    with open(filename, 'w') as f:
        f.write('len(r) =\n')
        f.write(f'{len(r)}\n')
        f.write('len(s) =\n')
        f.write(f'{len(s)}\n')
        f.write('len(np.unique(s0)) =\n')
        f.write(f'{len(np.unique(s0))}\n')
        f.write('tuple(len(ind) for ind in inds) =\n')
        f.write(f'{tuple(len(ind) for ind in inds)}\n')
        f.write('m =\n')
        f.write(f'{m}\n')
        f.write('ates =\n')
        f.write(f'({ates[0]:.4}, ')
        f.write(f'{ates[1]:.4}, ')
        f.write(f'{ates[2]:.4}, ')
        f.write(f'{ates[3]:.4})\n')
        f.write('ates / lenscales =\n')
        f.write(f'({ates[0] / lenscales[0]:.4}, ')
        f.write(f'{ates[1] / lenscales[1]:.4}, ')
        f.write(f'{ates[2] / lenscales[2]:.4}, ')
        f.write(f'{ates[3] / lenscales[2]:.4})\n')
        f.write('lenscales =\n')
        f.write(f'({lenscales[0]:.4}, ')
        f.write(f'{lenscales[1]:.4}, ')
        f.write(f'{lenscales[2]:.4})\n')
        f.write('kuipers =\n')
        f.write(f'({kuipers[0]:.4}, ')
        f.write(f'{kuipers[1]:.4}, ')
        f.write(f'{kuipers[2]:.4})\n')
        f.write('kolmogorov_smirnovs =\n')
        f.write(f'({kolmogorov_smirnovs[0]:.4}, ')
        f.write(f'{kolmogorov_smirnovs[1]:.4}, ')
        f.write(f'{kolmogorov_smirnovs[2]:.4})\n')
        f.write('kuiperss =\n')
        f.write(f'({kuiperss[0]:.4}, ')
        f.write(f'{kuiperss[1]:.4}, ')
        f.write(f'{kuiperss[2]:.4})\n')
        f.write('kolmogorov_smirnovss =\n')
        f.write(f'({kolmogorov_smirnovss[0]:.4}, ')
        f.write(f'{kolmogorov_smirnovss[1]:.4}, ')
        f.write(f'{kolmogorov_smirnovss[2]:.4})\n')
        f.write('kuipersp =\n')
        f.write(f'({kuipersp[0]:.4}, ')
        f.write(f'{kuipersp[1]:.4}, ')
        f.write(f'{kuipersp[2]:.4})\n')
        f.write('kolmogorov_smirnovsp =\n')
        f.write(f'({kolmogorov_smirnovsp[0]:.4}, ')
        f.write(f'{kolmogorov_smirnovsp[1]:.4}, ')
        f.write(f'{kolmogorov_smirnovsp[2]:.4})\n')
