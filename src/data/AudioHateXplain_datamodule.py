import json
import os

from datsetSplitter import createDatasetSplit, encodeData

class AudioHateXplainDataModule:
    def __init__(self, params, data_dir=None, dataset_name=None, cache_dir=None, token=None):
        self.params = params

        train, val, test = createDatasetSplit(self.params)

        self.dataset = {"train": train, "validation": val, "test": test}
    
    def get_dataset(self, params):
        train, val, test = createDatasetSplit(params)
        return train, val, test

