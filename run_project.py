from data_loader import data_loader
from pre_processor import pre_processor
from feature_extraction import FeatureExtractor
from dataset_generator import DataSplitter
from model_training import ModelTrainer
from evaluation import Evaluator

import numpy as np

if __name__ == "__main__":
    print("🚀 Starting Volatility Prediction Project...\n")

    # -----------------------------
    # 1. Load raw dataset
    # -----------------------------
    dataset = data_loader()   # مسیر داده را مشخص کن
    size, books = dataset.load()
    print("✅ Dataset loaded")
    print(f"   📊 Snapshots: {size}")
    print(f"   📐 Shape    : {books.shape}\n")

    # -----------------------------
    # 2. Preprocess data
    # -----------------------------
    preprocessor = pre_processor()
    clean_data = preprocessor.transform(books)

    # -----------------------------
    # 3. Feature extraction
    # -----------------------------
    extractor = FeatureExtractor(clean_data)
    features_dict = extractor.generate()
    print("✅ Feature extraction done")
    for name, values in features_dict.items():
        print(f"   🔹 {name} → shape: {values.shape}")

    # Combine all features into one matrix
    # حذف volatility از features
    feature_items = {k: v for k, v in features_dict.items() if k != "volatility"}
    features = np.column_stack(list(feature_items.values()))
    labels = features_dict["volatility"]

    print(f"\n📐 Final feature matrix: {features.shape}")
    print(f"🎯 Labels shape: {labels.shape}\n")

    # -----------------------------
    # 4. Split data
    # -----------------------------
    print("✂️ Splitting dataset...")
    splitter = DataSplitter(test_size=0.2, val_size=0.1)
    X_train, y_train, X_val, y_val, X_test, y_test = splitter.split(features, labels)
    print("✅ Data splitted")
    print(f"   🏋️ Train: {X_train.shape}")
    print(f"   🧪 Validation: {X_val.shape}")
    print(f"   🧾 Test: {X_test.shape}\n")

    # -----------------------------
    # 5. Train models (with tuning)
    # -----------------------------
    trainer = ModelTrainer()

    print("\n🔍 Running Hyperparameter Tuning...")

    rf_before, rf_after = trainer.tune_random_forest(X_train, y_train, X_val, y_val)
    xgb_before, xgb_after = trainer.tune_xgboost(X_train, y_train, X_val, y_val)

    tuning_results = [
        {"Model": "RandomForest", "MSE_before": rf_before, "MSE_after": rf_after},
        {"Model": "XGBoost", "MSE_before": xgb_before, "MSE_after": xgb_after},
    ]

    results = trainer.train_and_evaluate(X_train, y_train, X_val, y_val)

    # انتخاب بهترین مدل
    best_model, best_info = trainer.get_best_model(metric="mse")

    # ذخیره نتایج Validation به CSV
    df_val = trainer.results_to_dataframe(stage="validation")
    df_val.to_csv("validation_results.csv", index=False)

    # ذخیره مدل‌ها
    trainer.save_models("saved_models")

    # -----------------------------
    # 6. Evaluate results on Test
    # -----------------------------
    evaluator = Evaluator(results, tuning_results)
    test_results = evaluator.evaluate_on_test(X_test, y_test, plot=True)

    # ذخیره نتایج Test به CSV
    df_test = evaluator.test_results_to_dataframe()
    df_test.to_csv("test_results.csv", index=False)

    # رسم Residuals
    evaluator.plot_residuals(y_test)

    # -----------------------------
    # 7. Compare before/after tuning
    # -----------------------------
    evaluator.compare_tuning()

    print("\n🏁 Project pipeline finished.")
