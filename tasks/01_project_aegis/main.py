import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os
import warnings

# --- CONFIGURATION ---
MISSION_ID = "04_blind_search_final"
RESULTS_DIR = "results_blind_final"
POINTS_PER_CURVE = 500
TARGET_TOTAL_STARS = 500  
OUTLIER_FRACTION = 0.02

# THE SHOTGUN LIST: 
# 3 anchors in different sectors. If one fails, the next takes over.
ANCHOR_STARS = [
    "TIC 471016584",  # A dense field in Sector 1 (Southern Sky)
    "TIC 233681149",  # TRAPPIST-1 Field (Very famous, usually has data)
    "TIC 261136679"   # L 98-59 (Your "Lucky Star" backup)
]

def ensure_dirs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, RESULTS_DIR)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def fetch_and_process():
    all_data = []
    all_ids = []
    
    print(f"[-] [MISSION 4.1] INITIATING SHOTGUN SEARCH...")

    for anchor in ANCHOR_STARS:
        if len(all_data) >= TARGET_TOTAL_STARS: break
        
        needed = TARGET_TOTAL_STARS - len(all_data)
        
        print(f"\n[-] Moving telescope to field: {anchor}...")
        
        try:
            # We use a huge 3.0 degree radius to grab hundreds of neighbors
            search = lk.search_lightcurve(
                anchor, 
                radius=3.0, 
                limit=needed, 
                author="SPOC"
            )
            print(f"    > Locked on {len(search)} stars.")
            
            if len(search) == 0: 
                print("    [!] Sector empty or busy. Skipping...")
                continue

            lcs = search.download_all()
            if not lcs: continue

            print("    > Filtering & Processing...")
            
            for i, lc in enumerate(lcs):
                try:
                    if lc is None: continue
                    
                    try:
                        name = str(search.table[i]['target_name'])
                    except:
                        name = f"Star_{i}"

                    # --- CLEANING PIPELINE ---
                    lc = lc.remove_nans()
                    lc = lc.remove_outliers(sigma=3.5)
                    lc = lc.flatten(window_length=101)
                    lc = lc.normalize()
                    
                    duration = lc.time.max().value - lc.time.min().value
                    if duration <= 0: continue 
                    
                    bin_size = duration / POINTS_PER_CURVE
                    lc = lc.bin(time_bin_size=bin_size)
                    
                    flux = lc.flux.value
                    if len(flux) > POINTS_PER_CURVE:
                        flux = flux[:POINTS_PER_CURVE]
                    elif len(flux) < POINTS_PER_CURVE:
                        flux = np.pad(flux, (0, POINTS_PER_CURVE - len(flux)), constant_values=1.0)
                    
                    flux = np.nan_to_num(flux, nan=1.0)
                    
                    all_data.append(flux)
                    all_ids.append(name)
                    
                except:
                    continue
            
            print(f"    > Total Harvest: {len(all_data)} / {TARGET_TOTAL_STARS}")
            
        except Exception as e:
            print(f"    [!] Error in this sector: {e}")
            continue

    return np.array(all_data), all_ids

def run_ai_scan(X, ids, save_path):
    if len(X) == 0:
        print("[!] No data collected.")
        return

    print(f"\n[-] Training Isolation Forest on {len(X)} stars...")
    
    clf = IsolationForest(contamination=OUTLIER_FRACTION, random_state=42, n_jobs=-1)
    clf.fit(X)
    
    predictions = clf.predict(X)
    scores = clf.decision_function(X)
    
    anomalies_indices = np.where(predictions == -1)[0]
    scored_anomalies = sorted(zip(anomalies_indices, scores[anomalies_indices]), key=lambda x: x[1])
    
    print(f"[!] \033[92mSCAN COMPLETE. DETECTED {len(scored_anomalies)} CANDIDATES.\033[0m")
    
    for rank, (idx, score) in enumerate(scored_anomalies):
        star_id = ids[idx]
        
        plt.figure(figsize=(12, 5))
        plt.plot(X[idx], color='black', alpha=0.7, linewidth=0.8)
        
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
        plt.title(f"RANK #{rank+1}: {star_id} | Score: {score:.3f}")
        plt.xlabel("Normalized Time")
        plt.ylabel("Flux")
        
        safe_id = star_id.replace(" ", "_")
        filename = os.path.join(save_path, f"RANK_{rank+1}_{safe_id}.png")
        
        plt.savefig(filename)
        plt.close()
        print(f"    -> Saved: {filename}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    save_path = ensure_dirs()
    X, ids = fetch_and_process()
    run_ai_scan(X, ids, save_path)