import numpy as np
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, test_size=0.2, val_size=0.1, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split(self, features: np.ndarray, labels: np.ndarray):
        # تقسیم Train + Temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, 
            test_size=self.test_size + self.val_size, 
            shuffle=False  # چون داده سری زمانی هست
        )

        # تقسیم Validation و Test
        val_ratio = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=1 - val_ratio,
            shuffle=False
        )

        return X_train, y_train, X_val, y_val, X_test, y_test
