# scripts/create_labeled_dataset.py

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_financial_health_metrics(df, lookback_years=3):
    """
    Calculate financial health metrics for each company-year
    using lookback window of Y-3 to Y-1
    """
    results = []
    
    for company_id, company_data in df.groupby('company_id'):
        company_data = company_data.sort_values('financial_year_end')
        
        for idx, current_row in company_data.iterrows():
            current_year = current_row['financial_year_end']
            
            # Get lookback window: Y-3 to Y-1
            lookback_start = current_year - lookback_years
            lookback_end = current_year - 1
            
            lookback_data = company_data[
                (company_data['financial_year_end'] >= lookback_start) &
                (company_data['financial_year_end'] <= lookback_end)
            ]
            
            # Skip if we don't have enough lookback years
            if len(lookback_data) < lookback_years:
                continue
            
            # Calculate metrics from lookback period
            ocf_values = lookback_data['operating_cf'].dropna()
            
            if len(ocf_values) < 2:  # Need at least 2 values for volatility
                continue
            
            # 1. OCF positive count
            ocf_positive_count = (ocf_values > 0).sum()
            
            # 2. OCF volatility (standard deviation)
            ocf_volatility = ocf_values.std()
            
            # 3. Financing dependency flag
            # Check if company consistently relies on financing activities
            financing_cf_avg = lookback_data['financing_cf'].mean()
            financing_dependency = 1 if financing_cf_avg > 0 else 0
            
            # Assign label based on rules
            label = 0  # Default: Healthy
            
            # Risky conditions:
            # 1. More than half of OCF years are negative
            if ocf_positive_count < (len(ocf_values) / 2):
                label = 1
            # 2. High OCF volatility (more than 2x average OCF)
            elif ocf_volatility > (abs(ocf_values.mean()) * 2):
                label = 1
            # 3. Heavy financing dependency
            elif financing_dependency == 1 and financing_cf_avg > abs(ocf_values.mean()):
                label = 1
            
            result = {
                'company_id': company_id,
                'financial_year_end': current_year,
                'ocf_positive_count': ocf_positive_count,
                'ocf_volatility': ocf_volatility,
                'financing_dependency_flag': financing_dependency,
                'label': label,
                # Include original features for reference
                'operating_cf': current_row['operating_cf'],
                'investing_cf': current_row['investing_cf'],
                'financing_cf': current_row['financing_cf'],
                'total_assets': current_row['total_assets'],
                'revenue': current_row['revenue']
            }
            
            results.append(result)
    
    return pd.DataFrame(results)

def main():
    """Main function to create labeled dataset"""
    input_path = "canonical_financials.csv"
    output_path = "labeled_financials.csv"
    
    # Load canonical dataset
    print("Loading canonical financial data...")
    df = pd.read_csv(input_path)
    
    # Ensure proper sorting
    df = df.sort_values(['company_id', 'financial_year_end'])
    
    print(f"Original dataset: {len(df)} rows, {df['company_id'].nunique()} companies")
    
    # Calculate financial health metrics and labels
    print("Calculating financial health metrics and labels...")
    labeled_df = calculate_financial_health_metrics(df, lookback_years=3)
    
    # Filter out rows without sufficient lookback
    print(f"After lookback filtering: {len(labeled_df)} rows")
    
    # Check label distribution
    label_counts = labeled_df['label'].value_counts()
    print(f"Label distribution:\n{label_counts}")
    print(f"Healthy (0): {label_counts.get(0, 0)}")
    print(f"Risky (1): {label_counts.get(1, 0)}")
    
    # Save labeled dataset
    labeled_df.to_csv(output_path, index=False)
    print(f"Labeled dataset saved to: {output_path}")
    
    # Additional analysis
    print("\nAdditional analysis:")
    print(f"Companies with labeled data: {labeled_df['company_id'].nunique()}")
    print(f"Year range: {labeled_df['financial_year_end'].min()}–{labeled_df['financial_year_end'].max()}")
    
    # Show sample of risky companies
    risky_companies = labeled_df[labeled_df['label'] == 1]['company_id'].unique()
    print(f"Risky companies identified: {len(risky_companies)}")
    if len(risky_companies) > 0:
        print(f"Sample risky companies: {risky_companies[:5]}")

if __name__ == "__main__":
    main()