import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pymannkendall as mk

def generate_synthetic_data():
    """
    Generates synthetic groundwater level data that mimics the observed patterns.
    """
    # Create a date range from 2015 to 2024
    dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='MS')
    n = len(dates)

    # 1. Trend component (linearly decreasing)
    trend = -np.linspace(0, 5, n)

    # 2. Seasonal components (6-month, 12-month, 30-month cycles)
    t = np.arange(n)
    seasonality_12m = 2 * np.sin(2 * np.pi * t / 12)  # Annual cycle
    seasonality_6m = 0.5 * np.sin(2 * np.pi * t / 6)    # Semi-annual cycle
    seasonality_30m = 1.5 * np.sin(2 * np.pi * t / 30) # Longer term cycle

    # 3. Noise component
    noise = np.random.normal(0, 0.5, n)

    # Combine components
    level = 10 + trend + seasonality_12m + seasonality_6m + seasonality_30m + noise

    # Create DataFrame
    df = pd.DataFrame({'date': dates, 'level': level})
    df.set_index('date', inplace=True)

    return df

def impute_and_plot_data(df):
    """
    Introduces missing values, imputes them, and plots the results.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_missing = df.copy()

    # Introduce some missing values (e.g., 10% of the data)
    missing_indices = np.random.choice(df_missing.index, size=int(len(df_missing) * 0.1), replace=False)
    df_missing.loc[missing_indices, 'level'] = np.nan

    # Impute missing values using linear interpolation
    df_imputed = df_missing.interpolate(method='linear')

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df_imputed.index, df_imputed['level'], label='Imputed Data', color='blue', zorder=2)
    ax.scatter(df_missing.dropna().index, df_missing.dropna()['level'], label='Observed Data', color='red', s=20, zorder=3)

    # Highlight imputed sections
    is_imputed = df_missing['level'].isnull()
    for i, (date, imputed) in enumerate(is_imputed.items()):
        if imputed:
            ax.axvspan(date - pd.DateOffset(days=15), date + pd.DateOffset(days=15), color='grey', alpha=0.3, zorder=1)

    ax.set_title('Fig S1 (Recreated): Temporal Analysis of Groundwater Data')
    ax.set_xlabel('Year')
    ax.set_ylabel('Groundwater Level (m)')
    ax.legend()

    plt.savefig('Fig_S1_recreated.png')
    print("Recreated Figure S1 saved as 'Fig_S1_recreated.png'")

def perform_ssa_and_plot(df):
    """
    Performs a simplified Singular Spectrum Analysis (SSA) and plots the decomposition.
    """
    series = df['level']

    # 1. Embedding
    L = 36  # Window length (e.g., 3 years of months)
    N = len(series)
    K = N - L + 1
    X = np.zeros((L, K))
    for i in range(K):
        X[:, i] = series[i:i+L]

    # 2. SVD
    U, S, V = np.linalg.svd(X)

    # 3. Reconstruction (extracting the trend)
    # The first component is usually associated with the trend
    trend_component = S[0] * (U[:, 0].reshape(-1, 1) @ V[0, :].reshape(1, -1))

    # Diagonal averaging to convert back to a time series
    trend_series = np.zeros(N)
    for i in range(N):
        count = 0
        sum_val = 0
        for j in range(max(0, i - K + 1), min(i + 1, L)):
            sum_val += trend_component[j, i - j]
            count += 1
        trend_series[i] = sum_val / count if count > 0 else 0

    df['trend'] = trend_series
    df['residual'] = df['level'] - df['trend']

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # (a) Original Data
    axes[0].plot(df.index, df['level'], label='Original Time Series')
    axes[0].set_ylabel('Groundwater Level (m)')
    axes[0].set_title('(a) Data')
    axes[0].legend()

    # (b) Extracted Trend
    axes[1].plot(df.index, df['trend'], label='Extracted Trend', color='red')
    axes[1].set_ylabel('Trend (m)')
    axes[1].set_title('(b) SSA-Extracted Trend Component')
    axes[1].legend()

    # (c) Residual Noise
    axes[2].plot(df.index, df['residual'], label='Residual Noise', color='grey')
    axes[2].set_ylabel('Noise (m)')
    axes[2].set_title('(c) Residual Noise')
    axes[2].set_xlabel('Year')
    axes[2].legend()

    fig.suptitle('Fig S5 (Recreated): SSA Decomposition', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('Fig_S5_recreated.png')
    print("Recreated Figure S5 saved as 'Fig_S5_recreated.png'")


def plot_spectral_analysis(df):
    """
    Plots the power spectral density of the time series.
    """
    series = df['level'].values
    fs = 12  # 12 samples per year (monthly data)
    f, Pxx = signal.periodogram(series, fs)

    plt.figure(figsize=(10, 6))
    plt.semilogy(f, Pxx)
    plt.title('Fig S2 (Recreated): Periodogram of Groundwater Levels')
    plt.xlabel('Frequency (cycles/year)')
    plt.ylabel('Power Spectral Density')

    # Highlight significant periods from our synthetic data
    plt.axvline(x=1, color='red', linestyle='--', label='12 months')
    plt.axvline(x=2, color='red', linestyle='--', label='6 months')
    plt.axvline(x=12/30, color='red', linestyle='--', label='30 months')

    plt.savefig('Fig_S2_recreated.png')
    print("Recreated Figure S2 saved as 'Fig_S2_recreated.png'")

def perform_mk_test_and_plot(df):
    """
    Performs a Mann-Kendall test and creates a conceptual spatial plot.
    """
    # Perform MK test on the main time series
    result = mk.original_test(df['level'])
    print(f"Mann-Kendall test result for the main series: {result}")

    # Create a conceptual plot for spatial trends
    np.random.seed(42)
    num_points = 15
    lons = np.random.uniform(76, 80, num_points)
    lats = np.random.uniform(8, 14, num_points)

    # Assign trends conceptually
    trends = ['Declining (p<0.05)'] * 5 + ['Non-Significant'] * 10
    np.random.shuffle(trends)

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(num_points):
        if trends[i] == 'Declining (p<0.05)':
            ax.scatter(lons[i], lats[i], color='red', label='Declining (p<0.05)' if i == 0 else "")
        else:
            ax.scatter(lons[i], lats[i], color='grey', label='Non-Significant' if i == 5 else "")

    ax.set_title('Fig S4 (Recreated): Spatial Distribution of Mann-Kendall Trends')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()

    plt.savefig('Fig_S4_recreated.png')
    print("Recreated Figure S4 saved as 'Fig_S4_recreated.png'")


def main():
    """
    Main function to run the groundwater analysis.
    """
    print("Generating synthetic groundwater data...")
    gw_data = generate_synthetic_data()

    # --- Part 1: Imputation and Plotting ---
    impute_and_plot_data(gw_data.copy())

    # --- Part 2: SSA Decomposition ---
    perform_ssa_and_plot(gw_data.copy())

    # --- Part 3: Spectral Analysis ---
    plot_spectral_analysis(gw_data.copy())

    # --- Part 4: Mann-Kendall Trend Test ---
    perform_mk_test_and_plot(gw_data.copy())


if __name__ == "__main__":
    main()
