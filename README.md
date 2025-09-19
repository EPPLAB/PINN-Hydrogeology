# Groundwater Time Series Analysis

This project provides a Python script to analyze groundwater level time series data. It is a recreation of the analysis presented in a series of supplementary figures. Since the original data was not available, the script uses synthetically generated data that mimics the key characteristics of the original dataset (trend, seasonality, etc.).

## Features

The `groundwater_analysis.py` script performs the following analysis steps:

1.  **Synthetic Data Generation**: Creates a realistic groundwater time series with trend and seasonal components.
2.  **Imputation of Missing Data**: Simulates missing data points and fills them using linear interpolation.
3.  **Singular Spectrum Analysis (SSA)**: Decomposes the time series into its primary trend and residual components.
4.  **Spectral Analysis**: Calculates and plots the power spectral density of the time series to identify dominant periodicities.
5.  **Mann-Kendall Trend Test**: Performs a Mann-Kendall test to statistically identify the presence of a monotonic trend in the data.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the analysis script:**
    ```bash
    python groundwater_analysis.py
    ```

## Output

Running the script will generate the following output files, which are recreations of the original figures:

*   `Fig_S1_recreated.png`: Temporal analysis showing observed vs. imputed data.
*   `Fig_S2_recreated.png`: Periodogram showing the spectral analysis results.
*   `Fig_S4_recreated.png`: A conceptual map showing the spatial distribution of trends.
*   `Fig_S5_recreated.png`: SSA decomposition of the time series into trend and residuals.

The script will also print the results of the Mann-Kendall trend test to the console.
