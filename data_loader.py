from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from settings import DATASET_RAW_DATA_DIRECTORY, DATASET_RAW_DATA_FILE


class data_loader:
    def __init__(self):
        self.books = np.empty((0,), dtype=np.float32)
        self.size = 0

    def load(self):
        """Load ETH–USDT orderbook snapshots from JSONL file.
        Returns:
            tuple: (size, books) where 
                size  -> number of snapshots
                books -> numpy array of shape (size, 2, 2, N) [bids, asks]
        """
        file_path = Path(DATASET_RAW_DATA_DIRECTORY) / DATASET_RAW_DATA_FILE

        def extract(orders):
            return [o["price"] for o in orders], [o["quantity"] for o in orders]

        books = []
        with open(file_path) as f:
            for row in tqdm(f, desc="Loading orderbook"):
                try:
                    r = json.loads(row)
                    books.append([extract(r["bids"]), extract(r["asks"])])
                except Exception as e:
                    print(f"Error parsing row {len(books)}: {e}")
                    books.append(books[-1] if books else [([], []), ([], [])])

        self.books = np.array(books, dtype=np.float32)
        self.size = len(self.books)
        return self.size, self.books
