import os
from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxford_pets import OxfordPets

class Dalle_Flowers(DatasetBase):
    
    dataset_dir = 'dalle_oxford_flowers'

    def __init__(self, root, num_shots):
        # root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'jpg')
        self.split_path = os.path.join(self.dataset_dir, 'dalle_flower.json')

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        # train = self.generate_fewshot_dataset_noise(train, num_shots=num_shots,
        #                                                   dataset_name='dalle_OxfordFlowers')
        super().__init__(train_x=train, val=val, test=test)