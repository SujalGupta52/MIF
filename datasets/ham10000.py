import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxford_pets import OxfordPets


template = ["a photo of {}, a type of skin lesion."]


class HAM10000(DatasetBase):

    dataset_dir = "ham10000"

    def __init__(self, root, num_shots, kaggle_mode=False):
        self.dataset_dir = root

        if kaggle_mode:
            # In Kaggle, we'll use the paths as provided in the split file
            self.image_dir = (
                ""  # Base directory is root, paths in split file are relative
            )
            self.split_path = os.path.join("/kaggle/working", "split_HAM10000.json")
            if not os.path.exists(self.split_path):
                # Fall back to dataset directory if not found in working dir
                self.split_path = os.path.join(self.dataset_dir, "split_HAM10000.json")
        else:
            # Original implementation
            self.image_dir = os.path.join(self.dataset_dir, "HAM10000")
            self.split_path = os.path.join(self.dataset_dir, "split_HAM10000.json")

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train_cache = train.copy()
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        # Uncomment below if you want to generate noisy dataset
        # train_cache = self.generate_fewshot_dataset_noise(train_cache, num_shots=num_shots, dataset_name='HAM10000')

        super().__init__(train_x=train, val=val, test=test, train_cache=train_cache)
