import os

from .utils import Datum, DatasetBase
from .oxford_pets import OxfordPets


template = ['a photo of a {}.']


class Caltech101(DatasetBase):

    def __init__(self, root, num_shots):
        self.image_dir = os.path.join(root, '101_ObjectCategories')
        self.split_path = os.path.join(root, 'split_zhou_Caltech101.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train_cache = train.copy()
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        # train_cache = self.generate_fewshot_dataset_noise(train_cache, num_shots=num_shots, dataset_name='Caltech101')

        super().__init__(train_x=train, val=val, test=test ,train_cache = train_cache)