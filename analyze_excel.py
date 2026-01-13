import pandas as pd
import numpy as np
import os

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Sklearn not found, using simple binning/sampling.")

def analyze_data():
    file_path = "datos mittal.xlsx"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Read Excel
    print("Reading Excel...")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading excel: {e}")
        return

    print("Columns found:", df.columns.tolist())

    # Map meaningful columns (Case insensitive)
    # Expected: He_CaO, HE_SiO2, HE_MgO, He_FeO, He_Al2O3, HE_Bas (maybe MnO?)
    # Normalize cols
    df.columns = [c.strip() for c in df.columns]
    
    # Identify relevant columns
    # User image had specific names. Let's try to match them.
    relevant_cols = {
        'CaO': ['He_CaO', 'CaO', 'HE_CaO'],
        'SiO2': ['HE_SiO2', 'SiO2', 'He_SiO2'],
        'MgO': ['HE_MgO', 'MgO', 'He_MgO'],
        'FeO': ['He_FeO', 'FeO', 'HE_FeO'],
        'Al2O3': ['He_Al2O3', 'Al2O3', 'HE_Al2O3'],
        'MnO': ['MnO', 'He_MnO', 'HE_MnO'] # Might not exist
    }
    
    final_cols = {}
    for oxide, candidates in relevant_cols.items():
        for cand in candidates:
            # Case insensitive search
            match = next((c for c in df.columns if c.lower() == cand.lower()), None)
            if match:
                final_cols[oxide] = match
                break
    
    print("Mapped columns:", final_cols)

    if not final_cols:
        print("No relevant oxide columns found.")
        return

    # Create subset dataframe
    data = pd.DataFrame()
    for oxide, col_name in final_cols.items():
        data[oxide] = df[col_name]
    
    # Drop rows with NaN in critical columns
    data = data.dropna()
    print(f"Valid data rows: {len(data)}")
    
    if len(data) == 0:
        print("No valid data rows.")
        return

    # If MnO is missing, assume 0
    if 'MnO' not in data.columns:
        data['MnO'] = 0.0

    # Ensure all are numeric
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.dropna()

    # Get 10 representative points
    n_points = 10
    
    if HAS_SKLEARN and len(data) >= n_points:
        print("Using KMeans clustering...")
        kmeans = KMeans(n_clusters=n_points, random_state=42, n_init=10)
        kmeans.fit(data[['FeO', 'SiO2', 'CaO', 'MgO', 'Al2O3', 'MnO']])
        centroids = kmeans.cluster_centers_
        
        # Convert to dictionary list
        results = []
        for i, center in enumerate(centroids):
            # Center order matches input list
            # FeO, SiO2, CaO, MgO, Al2O3, MnO
            p = {
                'FeO': center[0],
                'SiO2': center[1],
                'CaO': center[2],
                'MgO': center[3],
                'Al2O3': center[4],
                'MnO': center[5],
                'label': f"Ind. Avg {i+1}"
            }
            results.append(p)
            
    else:
        print("Using random sampling (simplified)...")
        # Just sort by FeO to give a range
        data_sorted = data.sort_values('FeO')
        indices = np.linspace(0, len(data_sorted)-1, n_points, dtype=int)
        subset = data_sorted.iloc[indices]
        
        results = []
        for i, row in subset.iterrows():
            p = {
                'FeO': row['FeO'],
                'SiO2': row['SiO2'],
                'CaO': row['CaO'],
                'MgO': row['MgO'],
                'Al2O3': row['Al2O3'],
                'MnO': row['MnO'],
                'label': f"Ind. Sample {i+1}"
            }
            results.append(p)

    # Print results in Python format
    print("\nPROMEDIOS_INDUSTRIALES = [")
    for p in results:
        print(f"    {{'FeO': {p['FeO']:.2f}, 'SiO2': {p['SiO2']:.2f}, 'CaO': {p['CaO']:.2f}, 'MgO': {p['MgO']:.2f}, 'Al2O3': {p['Al2O3']:.2f}, 'MnO': {p['MnO']:.2f}, 'label': '{p['label']}'}},")
    print("]")

if __name__ == "__main__":
    analyze_data()
