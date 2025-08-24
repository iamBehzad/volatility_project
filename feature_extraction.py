"""
Feature Extraction Module
-------------------------
This module extracts meaningful features from the preprocessed order book data.

Implemented features:
1. Mid-Price
2. Price Change
3. Spread
4. Volatility
5. Bid Volume
6. Ask Volume
7. Volume Difference
"""

import numpy as np
import pandas as pd


class MidPriceExtractor:
    name = "mid_price"

    def __init__(self, dataset: np.ndarray) -> None:
        # mid price = (best ask price + best bid price) / 2
        self.feature_data = (dataset[:, 1, 0, 0] + dataset[:, 0, 0, 0]) / 2


class PriceChangeExtractor:
    name = "price_change"

    def __init__(self, mid_prices: np.ndarray) -> None:
        # percentage price change between snapshots
        self.feature_data = pd.Series(mid_prices).pct_change().fillna(0).values


class SpreadExtractor:
    name = "spread"

    def __init__(self, dataset: np.ndarray, mid_prices: np.ndarray) -> None:
        # spread = (best ask - best bid) / mid_price
        self.feature_data = np.abs(dataset[:, 1, 0, 0] - dataset[:, 0, 0, 0]) / mid_prices


class VolatilityExtractor:
    name = "volatility"

    def __init__(self, price_changes: np.ndarray, period: int = 60) -> None:
        # rolling variance of price changes
        number_of_samples = len(price_changes)
        volatilities = np.zeros((number_of_samples,))
        for i in range(number_of_samples):
            start = max(0, i - period)
            volatilities[i] = np.var(price_changes[start:i]) if i > 0 else 0
        self.feature_data = volatilities


class BidVolumeExtractor:
    name = "bid_volume"

    def __init__(self, dataset: np.ndarray) -> None:
        # sum of bid quantities across top depth
        self.feature_data = dataset[:, 0, 1, :].sum(axis=1)


class AskVolumeExtractor:
    name = "ask_volume"

    def __init__(self, dataset: np.ndarray) -> None:
        # sum of ask quantities across top depth
        self.feature_data = dataset[:, 1, 1, :].sum(axis=1)


class VolumeDifferenceExtractor:
    name = "volume_difference"

    def __init__(self, bid_volumes: np.ndarray, ask_volumes: np.ndarray) -> None:
        # absolute difference between total bid and ask volumes
        self.feature_data = np.abs(ask_volumes - bid_volumes)


class FeatureExtractor:
    """
    Combines all feature extractors into a single interface.
    """

    def __init__(self, dataset: np.ndarray):
        self.dataset = dataset

    def generate(self) -> dict:
        # 1. Mid price
        mid_price = MidPriceExtractor(self.dataset).feature_data

        # 2. Price change
        price_change = PriceChangeExtractor(mid_price).feature_data

        # 3. Spread
        spread = SpreadExtractor(self.dataset, mid_price).feature_data

        # 4. Volatility
        volatility = VolatilityExtractor(price_change, period=60).feature_data

        # 5. Bid & Ask volumes
        bid_volume = BidVolumeExtractor(self.dataset).feature_data
        ask_volume = AskVolumeExtractor(self.dataset).feature_data

        # 6. Volume difference
        volume_difference = VolumeDifferenceExtractor(bid_volume, ask_volume).feature_data

        # Return all features as a dictionary
        return {
            "mid_price": mid_price,
            "price_change": price_change,
            "spread": spread,
            "volatility": volatility,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "volume_difference": volume_difference,
        }
