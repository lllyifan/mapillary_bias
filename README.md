# Exploring geographic biases in Volunteered Street View Imagery: a case study of Mapillary data for England and Wales

## Reproduction Guide for Reviewers
To facilitate a step-by-step reproduction of the results presented in the paper, all Python scripts are named according to their corresponding figures in the manuscript.

### 1. Pre-requisite Analysis
Before generating the final figures, please run the following analysis scripts to compute the necessary indices and models. These scripts generate the intermediate data required for the plots.

| Target Figure | Run These First (Pre-requisites) | Purpose |
| :--- | :--- | :--- |
| **Fig 4** | `OCI_IA.py`, `OCI_IP.py` | Calculate the Over-concentration Index |
| **Fig 5** | `RF_IA.py`, `RF_IP.py`, `SHAP_IA.py`, `SHAP_IP.py` | Train RF models and generate SHAP explanations |
| **Fig 7** | `SHAP_INTERACTION_IA.py`, `SHAP_INTERACTION_IP.py` | Calculate SHAP interaction values |
| **Fig 8** | `ICE_IA.py`, `ICE_IP.py` | Calculate Individual Conditional Expectation (ICE) slopes |

### 2. Generating Figures
Once the pre-requisite scripts have been executed, you can run the scripts prefixed with `fig...` to produce the plots shown in the paper.

## Data 
This study draws on two primary data sources. The image dataset was taken from the Mapillary platform (https://www.mapillary.com/). The metadata were downloaded for all street-level images available for England and Wales up to 25th February 2025, yielding over 26 million (26170099) records. Detailed data is stored in ROOT/"data"/"raw"/"UK_mapillary-alltime_info_new.csv"

Demographic statistics were taken from the 2021 UK Census (https://www.ons.gov.uk/census), including the proportion of the neighbourhood population who are female (FP), the proportion of residents identifying with a ‘non-white’ ethnicity (NWP), the proportion aged 65 or older, who may be considered elderly (EP), the proportion of children, aged 15 or younger (CP), the proportion with no formal qualifications (NQP), the proportion unemployed (UEP), and the proportion of full-time students (SP). Detailed data is stored in ROOT/"data"/"raw"/"census2021.csv"
