"""
PreProcessor module
-------------------
This module cleans and normalizes the raw order book data.

Steps:
1. Keep only top-N levels of the order book.
2. Replace NaN or infinite values with 0.
3. Normalize prices and quantities using z-score normalization.
"""

import numpy as np


class pre_processor:
    def __init__(self, depth: int = 10):
        """
        Args:
            depth (int): number of top levels (bids/asks) to keep (default=10).
        """
        self.depth = depth

    def transform(self, books: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to order book snapshots.

        Args:
            books (np.ndarray): raw order book data (snapshots).

        Returns:
            np.ndarray: cleaned and normalized order book data.
        """
        # 1. Trim order book depth → only keep top-N levels
        books = books[:, :, :, :self.depth]

        # 2. Replace NaN or infinite values with 0
        books = np.nan_to_num(books, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Normalize with z-score (per snapshot)
        mean = books.mean(axis=(0, 3), keepdims=True)
        std = books.std(axis=(0, 3), keepdims=True) + 1e-8  # avoid division by zero
        books = (books - mean) / std

        return books
