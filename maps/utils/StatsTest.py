import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency, ttest_ind, anderson_ksamp

def identify_variable_type(series, unique_threshold=10):
    """
    Identifies the type of a variable in a pandas Series.

    Args:
        series (pd.Series): The data series to analyze.
        unique_threshold (int): The threshold for unique values to be
                                considered multiclass categorical instead of continuous.

    Returns:
        str: 'numerical', 'binary', or 'multiclass'.
    """
    num_unique = series.nunique()
    # Check for numeric types, but exclude boolean-like integers (0, 1) if it's binary
    if pd.api.types.is_numeric_dtype(series) and num_unique > 2 and num_unique > unique_threshold:
        return 'numerical'
    elif num_unique == 2:
        return 'binary'
    else:
        return 'multiclass'

def compare_distributions(real_df, raw_synth_df, refined_synth_df, alpha=0.05, numerical_test='ttest'):
    """
    Compares the distributions of variables across three dataframes (real, raw synthetic, refined synthetic).

    Args:
        real_df (pd.DataFrame): The dataframe with real data.
        raw_synth_df (pd.DataFrame): The dataframe with raw synthetic data.
        refined_synth_df (pd.DataFrame): The dataframe with refined synthetic data.
        alpha (float): The significance level for statistical tests.
        numerical_test (str): The statistical test for numerical data ('ttest', 'ks', or 'ad').

    Returns:
        pd.DataFrame: A dataframe with a multi-level header summarizing the statistics 
                      and test results for each variable.
    """
    results = []
    
    # --- Validate numerical_test parameter ---
    if numerical_test not in ['ttest', 'ks', 'ad']:
        raise ValueError("numerical_test must be either 'ttest', 'ks', or 'ad'")

    # --- Ensure all dataframes have the same columns ---
    if not (real_df.columns.equals(raw_synth_df.columns) and real_df.columns.equals(refined_synth_df.columns)):
        raise ValueError("Input DataFrames must have the same columns.")

    for col in real_df.columns:
        var_type = identify_variable_type(real_df[col])
        
        row = {'Variable Name': col, 'Type': var_type}

        # --- Extract series for easy access ---
        real_series = real_df[col].dropna()
        raw_series = raw_synth_df[col].dropna()
        refined_series = refined_synth_df[col].dropna()

        # --- Calculate Statistics and Perform Tests Based on Type ---
        if var_type == 'numerical':
            # --- Descriptive Statistics (Mean & Std) ---
            row['Real Data'] = f"{real_series.mean():.2f} (±{real_series.std():.2f})"
            row[('Raw Synthetic Data', 'Stat')] = f"{raw_series.mean():.2f} (±{raw_series.std():.2f})"
            row[('Refined Synthetic Data', 'Stat')] = f"{refined_series.mean():.2f} (±{refined_series.std():.2f})"
            
            # --- Statistical Test ---
            if numerical_test == 'ks':
                test_stat_raw, p_raw = ks_2samp(real_series, raw_series)
                test_stat_refined, p_refined = ks_2samp(real_series, refined_series)
            elif numerical_test == 'ad':
                ad_stat_raw, _, p_raw = anderson_ksamp([real_series, raw_series])
                ad_stat_refined, _, p_refined = anderson_ksamp([real_series, refined_series])
            else: # 'ttest'
                test_stat_raw, p_raw = ttest_ind(real_series, raw_series, equal_var=False) # Welch's t-test
                test_stat_refined, p_refined = ttest_ind(real_series, refined_series, equal_var=False)
            
            row[('Raw Synthetic Data', 'P-Value')] = f"{p_raw:.3f}"
            row[('Refined Synthetic Data', 'P-Value')] = f"{p_refined:.3f}"
            row[('Raw Synthetic Data', 'Match')] = 'Similar' if p_raw > alpha else 'Different'
            row[('Refined Synthetic Data', 'Match')] = 'Similar' if p_refined > alpha else 'Different'

        elif var_type == 'binary':
            # --- Descriptive Statistics (Proportion of class '1') ---
            row['Real Data'] = f"{real_series.value_counts(normalize=True).get(1, 0):.2%}"
            row[('Raw Synthetic Data', 'Stat')] = f"{raw_series.value_counts(normalize=True).get(1, 0):.2%}"
            row[('Refined Synthetic Data', 'Stat')] = f"{refined_series.value_counts(normalize=True).get(1, 0):.2%}"
            
            # --- Statistical Test (Chi-Squared) ---
            chi2_raw, p_raw, _, _ = chi2_contingency(pd.crosstab(real_series, raw_series))
            chi2_refined, p_refined, _, _ = chi2_contingency(pd.crosstab(real_series, refined_series))

            row[('Raw Synthetic Data', 'P-Value')] = f"{p_raw:.3f}"
            row[('Refined Synthetic Data', 'P-Value')] = f"{p_refined:.3f}"
            row[('Raw Synthetic Data', 'Match')] = 'Similar' if p_raw > alpha else 'Different'
            row[('Refined Synthetic Data', 'Match')] = 'Similar' if p_refined > alpha else 'Different'

        elif var_type == 'multiclass':
            # --- Descriptive Statistics (% of most frequent class) ---
            most_frequent_class = real_series.mode()[0]
            real_freq = real_series.value_counts(normalize=True).get(most_frequent_class, 0)
            raw_freq = raw_series.value_counts(normalize=True).get(most_frequent_class, 0)
            refined_freq = refined_series.value_counts(normalize=True).get(most_frequent_class, 0)

            row['Real Data'] = f"{real_freq:.2%} ({most_frequent_class})"
            row[('Raw Synthetic Data', 'Stat')] = f"{raw_freq:.2%}"
            row[('Refined Synthetic Data', 'Stat')] = f"{refined_freq:.2%}"

            # --- Statistical Test (Chi-Squared) ---
            chi2_raw, p_raw, _, _ = chi2_contingency(pd.crosstab(real_series, raw_series))
            chi2_refined, p_refined, _, _ = chi2_contingency(pd.crosstab(real_series, refined_series))
            
            row[('Raw Synthetic Data', 'P-Value')] = f"{p_raw:.3f}"
            row[('Refined Synthetic Data', 'P-Value')] = f"{p_refined:.3f}"
            row[('Raw Synthetic Data', 'Match')] = 'Similar' if p_raw > alpha else 'Different'
            row[('Refined Synthetic Data', 'Match')] = 'Similar' if p_refined > alpha else 'Different'

        results.append(row)

    # --- Create final DataFrame and build MultiIndex header ---
    temp_df = pd.DataFrame(results)
    temp_df.set_index(['Variable Name', 'Type', 'Real Data'], inplace=True)
    
    # Create the MultiIndex from the columns
    temp_df.columns = pd.MultiIndex.from_tuples(temp_df.columns)
    
    # Reset index to bring back 'Variable Name', 'Type', 'Real Data' as columns
    final_df = temp_df.reset_index()
    final_df.set_index('Variable Name', inplace=True)

    return final_df