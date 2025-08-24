import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import os


class ModelTrainer:
    def __init__(self):
        self.models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            "XGBoost": XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            ),
        }
        self.results = {}
        self.tuning_results = []  # ذخیره مقایسه قبل/بعد

    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        """
        آموزش همه مدل‌ها و ذخیره نتایج Validation
        """
        for name, model in self.models.items():
            print(f"\n🚀 Training {name} ...")
            model.fit(X_train, y_train)

            # پیش‌بینی روی Validation
            y_pred = model.predict(X_val)

            # محاسبه متریک‌ها
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            self.results[name] = {
                "model": model,
                "mse": mse,
                "mae": mae,
                "r2": r2,
            }

            print(f"✅ {name} trained")
            print(f"   📉 MSE: {mse:.6f}")
            print(f"   📉 MAE: {mae:.6f}")
            print(f"   📈 R2 : {r2:.6f}")

        return self.results

    def get_best_model(self, metric="mse"):
        """
        انتخاب بهترین مدل براساس یک متریک
        """
        if not self.results:
            raise ValueError("No models trained yet. Run train_and_evaluate first.")

        if metric in ["mse", "mae"]:
            best = min(self.results.items(), key=lambda x: x[1][metric])
        elif metric == "r2":
            best = max(self.results.items(), key=lambda x: x[1][metric])
        else:
            raise ValueError("Unsupported metric. Use 'mse', 'mae', or 'r2'.")

        name, info = best
        print(f"\n🏆 Best model based on {metric}: {name}")
        return info["model"], info

    def save_models(self, directory="saved_models"):
        """
        ذخیره همه مدل‌های آموزش‌دیده روی دیسک
        """
        os.makedirs(directory, exist_ok=True)
        for name, info in self.results.items():
            model_path = os.path.join(directory, f"{name}.pkl")
            joblib.dump(info["model"], model_path)
            print(f"💾 {name} model saved → {model_path}")

    def tune_random_forest(self, X_train, y_train, X_val, y_val):
        """
        Hyperparameter tuning for RandomForest (مقایسه روی Validation)
        """
        # مدل اولیه
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        base_model.fit(X_train, y_train)
        mse_before = mean_squared_error(y_val, base_model.predict(X_val))

        # جستجو
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
        }
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        mse_after = mean_squared_error(y_val, best_model.predict(X_val))

        print(f"\n🔧 Best RF Params: {grid_search.best_params_}")
        print(f"   📉 Best RF Score (Val MSE): {mse_after:.6f}")

        self.models["RandomForest"] = best_model
        self.tuning_results.append({
            "Model": "RandomForest",
            "MSE_before": mse_before,
            "MSE_after": mse_after
        })
        return mse_before, mse_after

    def tune_xgboost(self, X_train, y_train, X_val, y_val):
        """
        Hyperparameter tuning for XGBoost (مقایسه روی Validation)
        """
        base_model = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
        base_model.fit(X_train, y_train)
        mse_before = mean_squared_error(y_val, base_model.predict(X_val))

        param_dist = {
            "n_estimators": [200, 300, 500],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
        random_search = RandomizedSearchCV(
            XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
            param_distributions=param_dist, n_iter=10,
            scoring="neg_mean_squared_error", cv=3,
            n_jobs=-1, verbose=1, random_state=42
        )
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        mse_after = mean_squared_error(y_val, best_model.predict(X_val))

        print(f"\n🔧 Best XGB Params: {random_search.best_params_}")
        print(f"   📉 Best XGB Score (Val MSE): {mse_after:.6f}")

        self.models["XGBoost"] = best_model
        self.tuning_results.append({
            "Model": "XGBoost",
            "MSE_before": mse_before,
            "MSE_after": mse_after
        })
        return mse_before, mse_after

    def results_to_dataframe(self, stage="validation"):
        """
        نتایج مدل‌ها رو به DataFrame تبدیل می‌کنه
        """
        if not self.results:
            raise ValueError("No results available. Run train_and_evaluate first.")

        rows = []
        for name, info in self.results.items():
            rows.append({
                "Model": name,
                "MSE": info["mse"],
                "MAE": info["mae"],
                "R2": info["r2"],
            })

        df = pd.DataFrame(rows)
        print(f"\n📊 Model Performance ({stage})")
        print(df)
        return df
