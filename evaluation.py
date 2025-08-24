import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

class Evaluator:
    def __init__(self, models_results: dict, tuning_results=None):
        """
        models_results = {
            "LinearRegression": {"model": ..., "mse": ..., "mae": ..., "r2": ...},
            "RandomForest": {...},
            "XGBoost": {...}
        }
        tuning_results = [
            {"Model": "RandomForest", "MSE_before": ..., "MSE_after": ...},
            {"Model": "XGBoost", "MSE_before": ..., "MSE_after": ...}
        ]
        """
        self.models_results = models_results
        self.test_results = {}
        self.tuning_results = tuning_results

    def evaluate_on_test(self, X_test, y_test, plot=True):
        """
        ارزیابی مدل‌ها روی تست‌ست
        """
        for name, info in self.models_results.items():
            model = info["model"]
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            self.test_results[name] = {
                "mse": mse,
                "mae": mae,
                "mape": mape,
                "r2": r2,
                "y_pred": y_pred,
            }

            print(f"\n📊 Test Results - {name}")
            print(f"   📉 MSE : {mse:.6f}")
            print(f"   📉 MAE : {mae:.6f}")
            print(f"   📉 MAPE: {mape:.2f}%")
            print(f"   📈 R2  : {r2:.6f}")

        if plot:
            self._plot_results(y_test)

        return self.test_results

    def save_results_to_csv(self, filepath="test_results.csv"):
        if not self.test_results:
            raise ValueError("No test results available. Run evaluate_on_test first.")

        df = pd.DataFrame.from_dict(self.test_results, orient="index")
        df.to_csv(filepath)
        print(f"\n💾 Test results saved to {filepath}")
        return df

    def _plot_results(self, y_test, max_points=300):
        n = min(len(y_test), max_points)
        plt.figure(figsize=(12, 6))

        plt.plot(range(n), y_test[:n], label="True Volatility", color="black", linewidth=2)

        for name, info in self.test_results.items():
            plt.plot(range(n), info["y_pred"][:n], label=name, alpha=0.8)

        plt.title("Volatility Prediction - Test Set")
        plt.xlabel("Time step")
        plt.ylabel("Volatility")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_residuals(self, y_test, max_points=300):
        n = min(len(y_test), max_points)
        plt.figure(figsize=(12, 6))

        for name, info in self.test_results.items():
            residuals = y_test[:n] - info["y_pred"][:n]
            plt.plot(range(n), residuals, label=f"{name} Residuals", alpha=0.7)

        plt.axhline(0, color="black", linestyle="--")
        plt.title("Residuals (Prediction Errors)")
        plt.xlabel("Time step")
        plt.ylabel("Residual")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    def test_results_to_dataframe(self):
        if not self.test_results:
            raise ValueError("No test results available. Run evaluate_on_test first.")

        rows = []
        for name, info in self.test_results.items():
            row = {
                "Model": name,
                "MSE": info["mse"],
                "MAE": info["mae"],
                "MAPE": info["mape"],
                "R2": info["r2"],
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        print("\n📊 Test Set Performance")
        print(df)
        return df

    def compare_tuning(self):
        """
        مقایسه قبل و بعد از Hyperparameter Tuning
        """
        if not self.tuning_results:
            print("⚠️ No tuning results provided.")
            return None

        df = pd.DataFrame(self.tuning_results)

        print("\n📊 Hyperparameter Tuning Comparison")
        print(df)

        # نمودار مقایسه‌ای MSE
        df_melted = df.melt(id_vars=["Model"], value_vars=["MSE_before", "MSE_after"],
                            var_name="Stage", value_name="MSE")

        plt.figure(figsize=(8,5))
        for model in df["Model"]:
            subset = df_melted[df_melted["Model"] == model]
            plt.bar([f"{model}-{s}" for s in subset["Stage"]], subset["MSE"])
        plt.ylabel("MSE")
        plt.title("Model Performance Before vs After Hyperparameter Tuning")
        plt.xticks(rotation=30)
        plt.show()

        return df

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_idx = y_true != 0
    if not np.any(nonzero_idx):
        return np.nan
    return np.mean(np.abs((y_true[nonzero_idx] - y_pred[nonzero_idx]) / y_true[nonzero_idx])) * 100
