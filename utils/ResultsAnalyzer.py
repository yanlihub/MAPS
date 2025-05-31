import pandas as pd
import numpy as np
from IPython.display import display, HTML

def compare_synthcity_results(baseline_df, refined_df, display_table=True):
    """
    Compare synthcity evaluation results between baseline and refined synthetic data.
    
    Args:
        baseline_df (pd.DataFrame): Baseline evaluation results
        refined_df (pd.DataFrame): Refined evaluation results  
        display_table (bool): Whether to display the formatted table
        
    Returns:
        pd.DataFrame: Comparison table with better values highlighted
    """

    # Get the metric column name (first column)
    metric_col = baseline_df.columns[0]
    print(f"Metric column: {metric_col}")
    
    # Create comparison dataframe
    comparison_data = []
    
    # Process each metric in baseline
    for idx, row in baseline_df.iterrows():
        metric_full_name = row[metric_col]
        baseline_mean = row['mean']
        direction = row['direction']
        
        # Split metric name into class and name
        if '.' in metric_full_name:
            metric_class, metric_name = metric_full_name.split('.', 1)
        else:
            metric_class = 'other'
            metric_name = metric_full_name
            
        # Find corresponding metric in refined dataframe
        refined_row = refined_df[refined_df[metric_col] == metric_full_name]
        
        if not refined_row.empty:
            refined_mean = refined_row['mean'].iloc[0]
            
            comparison_data.append({
                'Metric Class': metric_class,
                'Metric Name': metric_name,
                'Baseline': baseline_mean,
                'Refined': refined_mean,
                'Direction': direction
            })
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison_data)
    
    if display_table:
        # Create styled version for display
        def highlight_better_value(row):
            """Apply styling to highlight better values"""
            styles = [''] * len(row)
            
            baseline_val = row['Baseline']
            refined_val = row['Refined']
            direction = row['Direction']
            
            if pd.isna(baseline_val) or pd.isna(refined_val):
                return styles
                
            if direction == 'maximize':
                if refined_val > baseline_val:
                    styles[3] = 'font-weight: bold; color: green'  # Refined column
                elif baseline_val > refined_val:
                    styles[2] = 'font-weight: bold; color: green'  # Baseline column
            elif direction == 'minimize':
                if refined_val < baseline_val:
                    styles[3] = 'font-weight: bold; color: green'  # Refined column
                elif baseline_val < refined_val:
                    styles[2] = 'font-weight: bold; color: green'  # Baseline column
                    
            return styles
        
        # Apply styling
        styled_df = comparison_df.style.apply(highlight_better_value, axis=1)
        
        # Format numeric columns
        styled_df = styled_df.format({
            'Baseline': '{:.6f}',
            'Refined': '{:.6f}'
        })
        
        print("=" * 80)
        print("SYNTHCITY EVALUATION COMPARISON")
        print("=" * 80)
        print("Green bold values indicate better performance")
        print("=" * 80)
        
        display(styled_df)
        
        # Print summary statistics
        print("\nSUMMARY:")
        
        # Count improvements
        improvements = 0
        degradations = 0
        unchanged = 0
        
        for _, row in comparison_df.iterrows():
            baseline_val = row['Baseline']
            refined_val = row['Refined']
            direction = row['Direction']
            
            if pd.isna(baseline_val) or pd.isna(refined_val):
                continue
                
            if direction == 'maximize':
                if refined_val > baseline_val:
                    improvements += 1
                elif refined_val < baseline_val:
                    degradations += 1
                else:
                    unchanged += 1
            elif direction == 'minimize':
                if refined_val < baseline_val:
                    improvements += 1
                elif refined_val > baseline_val:
                    degradations += 1
                else:
                    unchanged += 1
        
        total_metrics = len(comparison_df)
        print(f"Total metrics: {total_metrics}")
        print(f"Improvements: {improvements} ({improvements/total_metrics*100:.1f}%)")
        print(f"Degradations: {degradations} ({degradations/total_metrics*100:.1f}%)")
        print(f"Unchanged: {unchanged} ({unchanged/total_metrics*100:.1f}%)")
        
        # Calculate average improvement by metric class
        print("\nIMPROVEMENT BY METRIC CLASS:")
        for metric_class in comparison_df['Metric Class'].unique():
            class_df = comparison_df[comparison_df['Metric Class'] == metric_class]
            class_improvements = 0
            class_total = 0
            
            for _, row in class_df.iterrows():
                baseline_val = row['Baseline']
                refined_val = row['Refined']
                direction = row['Direction']
                
                if pd.isna(baseline_val) or pd.isna(refined_val):
                    continue
                    
                class_total += 1
                if direction == 'maximize' and refined_val > baseline_val:
                    class_improvements += 1
                elif direction == 'minimize' and refined_val < baseline_val:
                    class_improvements += 1
            
            if class_total > 0:
                improvement_rate = class_improvements / class_total * 100
                print(f"  {metric_class}: {class_improvements}/{class_total} improved ({improvement_rate:.1f}%)")
    
    return comparison_df


def create_detailed_comparison_table(baseline_df, refined_df):
    """
    Create a more detailed comparison table with additional statistics.
    
    Args:
        baseline_df (pd.DataFrame): Baseline evaluation results
        refined_df (pd.DataFrame): Refined evaluation results
        
    Returns:
        pd.DataFrame: Detailed comparison table
    """
    
    # Get the metric column name (first column)
    metric_col = baseline_df.columns[0]
    
    detailed_data = []
    
    for idx, baseline_row in baseline_df.iterrows():
        metric_full_name = baseline_row[metric_col]
        
        # Split metric name
        if '.' in metric_full_name:
            metric_class, metric_name = metric_full_name.split('.', 1)
        else:
            metric_class = 'other'
            metric_name = metric_full_name
            
        # Find corresponding refined metric
        refined_row = refined_df[refined_df[metric_col] == metric_full_name]
        
        if not refined_row.empty:
            refined_row = refined_row.iloc[0]
            
            # Calculate improvement
            baseline_mean = baseline_row['mean']
            refined_mean = refined_row['mean']
            direction = baseline_row['direction']
            
            if direction == 'maximize':
                improvement = refined_mean - baseline_mean
                improvement_pct = (improvement / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
                is_better = refined_mean > baseline_mean
            else:  # minimize
                improvement = baseline_mean - refined_mean
                improvement_pct = (improvement / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
                is_better = refined_mean < baseline_mean
            
            detailed_data.append({
                'Metric Class': metric_class,
                'Metric Name': metric_name,
                'Baseline Mean': baseline_mean,
                'Baseline Std': baseline_row['stddev'],
                'Refined Mean': refined_mean,
                'Refined Std': refined_row['stddev'],
                'Absolute Improvement': improvement,
                'Percentage Improvement': improvement_pct,
                'Direction': direction,
                'Is Better': is_better
            })
    
    return pd.DataFrame(detailed_data)