import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os

# --- CONFIGURATION ---
MISSION_ID = "01_project_aegis"
RESULTS_DIR = "results"  # Will be created inside the project folder
NUM_STARS = 50           # Start with 50 stars for a quick test
OUTLIER_FRACTION = 0.05  # Top 5% weirdest stars

def ensure_dirs():
    """Creates the results directory if it doesn't exist."""
    path = os.path.join(os.path.dirname(__file__), RESULTS_DIR)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def fetch_and_process(num_stars):
    print(f"[-] [A.E.G.I.S] Downloading {num_stars} stars from TESS Sector 14...")
    # Search for stars
    search = lk.search_lightcurve("TESS", sector=14, limit=num_stars)
    
    # Download
    lcs = search.download_all()
    print("[-] Download complete. Processing...")

    data_matrix = []
    valid_ids = []

    for i, lc in enumerate(lcs):
        try:
            # Clean and Normalize
            lc = lc.remove_nans().normalize().flatten()
            
            # Binning: Force all curves to be 500 points long for the AI
            lc = lc.bin(time_bin_size=(lc.time.max() - lc.time.min()) / 500)
            
            # Extract Flux
            flux = lc.flux.value
            if len(flux) == 500: # strict check
                # Fill missing data with 1.0 (flat line)
                flux = np.nan_to_num(flux, nan=1.0)
                data_matrix.append(flux)
                valid_ids.append(search.table[i]['target_name'])
        except:
            continue
            
    return np.array(data_matrix), valid_ids

def run_ai_scan(X, ids, save_path):
    print("[-] Training Isolation Forest...")
    clf = IsolationForest(contamination=OUTLIER_FRACTION, random_state=42)
    clf.fit(X)
    predictions = clf.predict(X) # -1 is anomaly
    
    anomalies = np.where(predictions == -1)[0]
    print(f"[!] DETECTED {len(anomalies)} ANOMALIES.")

    for idx in anomalies:
        star_id = ids[idx]
        plt.figure(figsize=(10, 4))
        plt.plot(X[idx], color='red', alpha=0.7)
        plt.title(f"ANOMALY DETECTED: {star_id}")
        plt.xlabel("Normalized Time")
        plt.ylabel("Flux")
        
        # Save to the results folder
        filename = os.path.join(save_path, f"anomaly_{star_id}.png")
        plt.savefig(filename)
        plt.close()
        print(f"    -> Saved report: {filename}")

if __name__ == "__main__":
    save_path = ensure_dirs()
    X, ids = fetch_and_process(NUM_STARS)
    
    if len(X) > 0:
        run_ai_scan(X, ids, save_path)
    else:
        print("[!] Data download failed. Check internet.")