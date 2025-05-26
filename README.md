# Electricity‑Consumption‑Forecast

Forecasting electricity consumption in France, using RTE, weather and calendar data.
This project provides a complete pipeline: data extraction and cleaning, EDA, feature engineering, residual modeling via XGBoost, and evaluation.

## Table of Contents

- [Installation](#installation)
- [Content](#content)
- [Usage](#usage)
- [Contact](#contact)

## Installation

Download the repository and install the conda environment using the following bash command:

```conda env create -f Electricity-consumption-in-France.yml```

## Content

### Data ingestion

* Load RTE consumption series

* Load daily departmental temperatures (Data‑Gouv)

* Bank holidays, school vacations, TEMPO day types

### Exploratory Data Analysis (EDA)

* Daily and annual seasonality

* Weekday/weekend/vacation/holiday effects

* Correlation analysis: consumption vs HDD/CDD

### Preprocessing 

#### Feature engineering

* HDD, CDD (Heating/Cooling Degree Days)

* hour_sin, hour_cos (daily cycle)

* day_of_week (0–6)

* lag_24h, roll_mean_7d (autocorrelation & trend)

* resid_j‑1 (naive error from previous day)

* Holiday, vacation, TEMPO flags (one‑hot)

#### Feature imputing / scaling / enconding

* Imputation: mean+indicator for HDD/CDD, constant “BLEU” for TEMPO

* Scaling: StandardScaler for continuous features

* Encoding: OneHotEncoder(drop='first') for day_of_week & TEMPO

### Modeling

* Naive persistence baseline

* XGBoost on day‑ahead residual

* Time‑series validation split (train/valid/test)
  
* Metrics : MAE, RMSE

## Usage

Usage Instructions (in this order):
* Run ```python extract_and_combine_data_sources.py``` : extract the data and save the DataFrame in final_df.csv.
* Run the notebook ```data_analysis.ipynb``` : perform and display the EDA.
* Run the notebook ```preprocessing and_training.ipynb``` : perform the preprocessing and the training, and display the results.

The notebook ```various_check.ipynb``` contains only small analyses and checks.

## Results

See the PDF "Electricity consumption in France". Figures and graphs in English but texts in French.

## Contact

If you have any questions or suggestions, please contact :

Rémi Dromnelle — remi.dromnelle@gmail.com
