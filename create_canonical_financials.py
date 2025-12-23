# scripts/create_canonical_financials.py

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re

def normalize_financial_year(year_str):
    """
    Convert year string to canonical financial year end integer
    Rule: Mar YYYY → YYYY, Dec YYYY → YYYY + 1
    """
    if pd.isna(year_str) or year_str is None:
        return None
        
    year_str = str(year_str).strip()
    
    # Match patterns like "Mar 2014", "Dec 2012", "2014", "2024.5"
    mar_match = re.match(r'Mar\s+(\d{4})', year_str, re.IGNORECASE)
    dec_match = re.match(r'Dec\s+(\d{4})', year_str, re.IGNORECASE)
    year_match = re.match(r'(\d{4})(\.\d+)?$', year_str)
    
    if mar_match:
        return int(mar_match.group(1))
    elif dec_match:
        return int(dec_match.group(1)) + 1
    elif year_match:
        return int(year_match.group(1))
    else:
        return None

def process_company_json(file_path):
    """Process a single company JSON file and return canonical financial data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    company_id = data['company']['id']
    company_meta = {
        'sector': 'Unknown',  # Would need to be added to company data
        'face_value': safe_float(data['company'].get('face_value')),
        'book_value': safe_float(data['company'].get('book_value')),
        'roe_trailing': safe_float(data['company'].get('roe_percentage')),
        'roce_trailing': safe_float(data['company'].get('roce_percentage'))
    }
    
    # Extract and normalize all financial data by year
    financial_data = {}
    
    # Process cashflow data
    for cf in data['data'].get('cashflow', []):
        year_end = normalize_financial_year(cf.get('year'))
        if year_end is None:
            continue
            
        if year_end not in financial_data:
            financial_data[year_end] = {'financial_year_end': year_end}
            
        financial_data[year_end].update({
            'operating_cf': safe_float(cf.get('operating_activity')),
            'investing_cf': safe_float(cf.get('investing_activity')),
            'financing_cf': safe_float(cf.get('financing_activity')),
            'net_cf': safe_float(cf.get('net_cash_flow'))
        })
    
    # Process balance sheet data
    for bs in data['data'].get('balancesheet', []):
        year_end = normalize_financial_year(bs.get('year'))
        if year_end is None:
            continue
            
        if year_end not in financial_data:
            financial_data[year_end] = {'financial_year_end': year_end}
            
        financial_data[year_end].update({
            'total_assets': safe_float(bs.get('total_assets')),
            'total_liabilities': safe_float(bs.get('total_liabilities')),
            'equity': safe_float(bs.get('reserves')) + safe_float(bs.get('equity_capital', 0)),
            'borrowings': safe_float(bs.get('borrowings'))
        })
    
    # Process profit and loss data
    for pl in data['data'].get('profitandloss', []):
        year_end = normalize_financial_year(pl.get('year'))
        if year_end is None:
            continue
            
        if year_end not in financial_data:
            financial_data[year_end] = {'financial_year_end': year_end}
            
        financial_data[year_end].update({
            'revenue': safe_float(pl.get('sales')),
            'profit_after_tax': safe_float(pl.get('net_profit'))
        })
    
    # Add company metadata and filter out invalid rows
    rows = []
    for year_data in financial_data.values():
        # Skip TTM and partial data
        if year_data.get('financial_year_end') is None:
            continue
            
        # Skip zero-filled financial statements
        if (safe_float(year_data.get('operating_cf')) == 0 and 
            safe_float(year_data.get('investing_cf')) == 0 and 
            safe_float(year_data.get('financing_cf')) == 0):
            continue
            
        row = {
            'company_id': company_id,
            'financial_year_end': year_data['financial_year_end'],
            'operating_cf': year_data.get('operating_cf'),
            'investing_cf': year_data.get('investing_cf'),
            'financing_cf': year_data.get('financing_cf'),
            'net_cf': year_data.get('net_cf'),
            'total_assets': year_data.get('total_assets'),
            'total_liabilities': year_data.get('total_liabilities'),
            'equity': year_data.get('equity'),
            'revenue': year_data.get('revenue'),
            'profit_after_tax': year_data.get('profit_after_tax')
        }
        row.update(company_meta)
        rows.append(row)
    
    return rows

def safe_float(val):
    """Safely convert value to float, return None if conversion fails"""
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def main():
    """Main function to create canonical financial dataset"""
    raw_data_path = Path("data/raw")
    output_path = "canonical_financials.csv"
    
    all_rows = []
    company_count = 0
    
    print("Processing company JSON files...")
    
    for json_file in raw_data_path.glob("*.json"):
        company_count += 1
        print(f"Processing {json_file.name}...")
        
        rows = process_company_json(json_file)
        if rows:
            all_rows.extend(rows)
    
    if not all_rows:
        print("No data processed!")
        return
    
    # Create DataFrame and apply final checks
    df = pd.DataFrame(all_rows)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['company_id', 'financial_year_end'])
    
    # Sort by company and year
    df = df.sort_values(['company_id', 'financial_year_end'])
    
    # Data quality checks
    print("\nRunning sanity checks...")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['company_id', 'financial_year_end']).sum()
    print(f"Duplicate rows found: {duplicates}")
    
    # Check year integrity per company
    year_integrity_issues = 0
    for company_id, company_data in df.groupby('company_id'):
        years = sorted(company_data['financial_year_end'].tolist())
        if years != sorted(set(years)):  # Check for duplicates
            year_integrity_issues += 1
        if len(years) < 5:
            print(f"Company {company_id} has only {len(years)} years of data")
    
    print(f"Year integrity issues: {year_integrity_issues}")
    
    # Check numeric values
    numeric_cols = ['operating_cf', 'investing_cf', 'financing_cf', 'net_cf', 
                   'total_assets', 'total_liabilities', 'equity', 'revenue', 'profit_after_tax']
    for col in numeric_cols:
        non_numeric = df[col].apply(lambda x: not pd.isna(x) and not isinstance(x, (int, float, np.number))).sum()
        if non_numeric > 0:
            print(f"Non-numeric values in {col}: {non_numeric}")
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"\nCanonical dataset created successfully!")
    print(f"Rows: {len(df)}")
    print(f"Companies: {df['company_id'].nunique()}")
    print(f"Year range: {df['financial_year_end'].min()}–{df['financial_year_end'].max()}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()