# 📈 Volatility Prediction Project

A machine learning pipeline for **predicting market volatility** using Ethereum Order Book data.  
This project implements data preprocessing, feature extraction, model training (Linear Regression, Random Forest, and XGBoost), hyperparameter tuning, and evaluation on a 3-month dataset.  

🔗 **Project Repository:** [volatility_project](https://github.com/iamBehzad/volatility_project)

---

## 🚀 Project Overview
Volatility forecasting is crucial for **risk management** and **decision-making** in financial markets.  
This project uses **Ethereum order book snapshots** to extract features and predict short-term volatility.

Key stages of the pipeline:
1. **Data Preparation** → Load and preprocess raw order book data (cleaning, ordering, normalization).
2. **Feature Engineering** → Extract features such as mid-price, price change, spread, bid/ask volumes, volume imbalance, etc.
3. **Target Variable (Volatility)** → Computed based on price changes using a rolling window.
4. **Dataset Splitting** → Train (70%), Validation (10%), and Test (20%) with time-series consistency (`shuffle=False`).
5. **Model Training** → Linear Regression (baseline), RandomForest, XGBoost.
6. **Hyperparameter Tuning** → GridSearchCV / RandomizedSearchCV for RF and XGB.
7. **Evaluation** → Metrics (MSE, MAE, R², MAPE), prediction plots, residual plots, tuning comparisons.
8. **Model Saving** → Trained models are stored as `.pkl` files.

---

## 📂 Repository Structure
```
volatility_project/
│
├── data_loader.py          # Load raw dataset
├── pre_processor.py        # Data cleaning & preprocessing
├── feature_extraction.py   # Feature engineering
├── dataset_generator.py    # Train/Val/Test splitting
├── model_training.py       # Model training & hyperparameter tuning
├── evaluation.py           # Evaluation metrics & visualization
├── run_project.py          # Main pipeline script
│
├── saved_models/           # Saved trained models (.pkl files)
├── validation_results.csv  # Validation set metrics
├── test_results.csv        # Test set metrics
│
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/iamBehzad/volatility_project.git
   cd volatility_project
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # (Linux/Mac)
   .venv\Scripts\activate      # (Windows)
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

Run the full pipeline with:
```bash
python run_project.py
```

Outputs:
- **Console logs**: Detailed step-by-step pipeline execution.  
- **CSV files**: Validation & Test results.  
- **Plots**: Model predictions vs. true values, residual plots, tuning comparison.  
- **Saved models**: Exported to `saved_models/` as `.pkl`.  

---

## 📊 Results Summary

Validation and test results showed the following trends:
- **Linear Regression** → Surprisingly performed best on MSE (serving as a strong baseline).
- **Random Forest & XGBoost** → Improved after hyperparameter tuning, but struggled with non-stationary behavior in volatility.
- **MAPE values** → Extremely high due to near-zero volatility targets (relative error instability).

Hyperparameter tuning comparison:
| Model         | MSE (Before) | MSE (After) |
|---------------|--------------|-------------|
| Random Forest | 251.68       | 170.09      |
| XGBoost       | 304.58       | 118.97      |

---

## 📈 Visualizations
The project produces:
- Predictions vs. True Volatility (line plots).
- Residual plots for error analysis.
- Bar charts comparing model performance before/after hyperparameter tuning.

---

## 📝 Future Work
- Introduce **deep learning models** (e.g., LSTM, Transformer) for sequential patterns.
- Incorporate **longer historical datasets**.
- Add **cross-market indicators** (BTC, global crypto volume, etc.).
- Enhance feature set with **rolling statistics** and **order book imbalance indicators**.

---

## 👤 Author
- **Behzad**  
🔗 [GitHub Profile](https://github.com/iamBehzad)

---

## 📜 License
This project is licensed under the **MIT License** – you are free to use, modify, and distribute with attribution.
