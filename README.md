# NFT & Mt. Gox Wash Trading ML Analysis

This repository contains code for machine learning analysis of wash trading in NFT marketplaces and the Mt. Gox exchange.

## Folder Structure

- `data/`  
  - **Not included:** Raw labeled datasets and generated ML samples (too large).  
    - Inputs:  
      - `data/filtered/blur_sales_filtered.csv`  
      - `data/filtered/looksrare_sales_filtered.csv`  
      - `data/filtered/opensea_sales_filtered.csv`  
      - `data/filtered/gox_sales_labeled.csv`  
    - Outputs:  
      - `data/ml_sample/Blur_ml_samples.csv`  
      - `data/ml_sample/LooksRare_ml_samples.csv`  
      - `data/ml_sample/OpenSea_ml_samples.csv`  
      - `data/ml_sample/gox_ml_samples.csv`  

  - **Included:** Final ML outputs in `data/results/`:  
    - `nft_ml_feature_importance.csv`  
    - `nft_ml_metrics.csv`  
    - `gox_ml_feature_importance.csv`  
    - `gox_ml_metrics.csv`  

- `scripts/`  
  - `*_ml_sample.py`: Converts filtered, tagged datasets into ML-ready samples (Step 1).  
  - `*_ml_run.py`: Applies ML models to the prepared samples (Step 2).  
  - `results_analysis.py`: Summarizes model performance and feature importance.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Step 1: Prepare ML samples

python scripts/nft_ml_sample.py
python scripts/gox_ml_sample.py

Step 2: Run ML models

python scripts/nft_ml_run.py
python scripts/gox_ml_run.py

Step 3: Analyze results

python scripts/results_analysis.py

## Contact

For access to the tagged datasets or ML samples, please contact Niuniu Zhang at [niuniu.zhang.phd@anderson.ucla.edu](mailto:niuniu.zhang.phd@anderson.ucla.edu) or [niuniu@ucla.edu](mailto:niuniu@ucla.edu).
