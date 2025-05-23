import os
import random
import os.path as osp
import tarfile
import zipfile
import gdown
import numpy as np
import pickle
import json
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from collections import defaultdict
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _train_preprocess(size=224, scale=(0.8, 1)):
    funcs = [
        # T.RandomRotation(10),
        T.RandomResizedCrop(size=size, scale=scale, interpolation=BICUBIC),
        # T.AutoAugment()
        T.RandomHorizontalFlip(p=0.5),
        # T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
    ]
    return funcs


def _test_preprocess(size):
    funcs = [
        Resize(size, interpolation=BICUBIC),
        CenterCrop(size),
    ]
    return funcs


def _basic_postprocess():
    funcs = [
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]
    return funcs


def _transform(size, is_train=True, **args):
    preprocess = _train_preprocess(size, **args) if is_train else _test_preprocess(size)
    postprocess = _basic_postprocess()
    return Compose(preprocess + postprocess)
    

class _patch_transform:
    
    def __init__(self, size, n, m, is_train=True) -> None:
        self.n = n
        self.m = m
        preprocess = _train_preprocess(size * n) if is_train else _test_preprocess(size * n)
        postprocess = _basic_postprocess()
        self.pre = Compose(preprocess + postprocess)
    
    def __call__(self, img):
        img = self.pre(img)
        _, height, width = img.size()
        piece_height = height // self.n
        piece_width = width // self.m
        pieces = []
        for i in range(self.n):
            for j in range(self.m):
                piece = img[:, i * piece_height:(i + 1) * piece_height, j * piece_width:(j + 1) * piece_width]
                pieces.append(piece)
        return pieces


class _randcrop_transform:
    
    def __init__(self, patch: int, size=224, scale=(0.2, 1), is_train=True) -> None:
        self.patch = patch
        pre = [T.RandomResizedCrop(size=size, scale=scale, interpolation=BICUBIC)]
        if is_train:
            ot_aug = T.RandomHorizontalFlip(p=0.5)
            pre.append(ot_aug)
        postprocess = _basic_postprocess()
        self.transform = Compose(pre + postprocess)
        
    def __call__(self, img):
        imgs = [self.transform(img) for _ in range(self.patch)]
        return imgs


class _5crop_transform:
    
    def __init__(self, size=224, is_train=True) -> None:
        pre = [Resize(size=size, interpolation=BICUBIC)]
        if is_train:
            ot_aug = T.RandomHorizontalFlip(p=0.5)
            pre.append(ot_aug)
        postprocess = _basic_postprocess()
        self.transform = Compose(pre + postprocess)
        
    def __call__(self, img):
        s = min(img.size) // 2
        crops = F.five_crop(img, size=s)
        imgs = [self.transform(crop) for crop in crops]
        return imgs


class _gradcam_transform:
    
    def __init__(self, size=224) -> None:
        trans = _train_preprocess(size) + _basic_postprocess()[:-1]
        self.transform = Compose(trans)
        
    def __call__(self, img):
        return [self.transform(img)]
    
    
# (1 + p) * batch_size
def _my_collate(batch):
    global_img = []
    local_img = []
    labels = []
    for img, label in batch:
        global_img.append(img[0])
        local_img += img[1:]
        labels.append(label)
    imgs = torch.stack(global_img + local_img)
    return imgs, torch.tensor(labels)



# followings are inherited from Tip-Adapter

def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not osp.exists(path):
        raise IOError('No file exists at {}'.format(path))

    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(
                'Cannot read image from {}, '
                'probably due to heavy IO. Will re-try'.format(path)
            )


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith('.') and 'sh' not in f]
    if sort:
        items.sort()
    return items


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath='', label=0, domain=-1, classname=''):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """
    dataset_dir = '' # the directory where the dataset is stored
    domains = [] # string names of all domains

    def __init__(self, train_x=None, train_u=None, val=None, test=None,train_cache = None):
        self._train_x = train_x # labeled training data
        self._train_u = train_u # unlabeled training data (optional)
        self._val = val # validation data (optional)
        self._test = test # test data
        self._train_cache = train_cache
        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_cache(self):
        return self._train_cache

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    'Input domain must belong to {}, '
                    'but got [{}]'.format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print('Extracting file ...')

        try:
            tar = tarfile.open(dst)
            tar.extractall(path=osp.dirname(dst))
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(dst, 'r')
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        print('File extracted to {}'.format(osp.dirname(dst)))

    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=True
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed.
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot dataset')

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output


    def generate_fewshot_dataset_noise(
        self, *data_sources, num_shots=-1, repeat=True , dataset_name = ''
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed.
        """

        def montecarlo_selection_by_class(sampled_items, num_samples=10, shots=16):
            """从sampled_items列表中选出shots个最不熟悉的样本"""

            def  gaussian_noise(image, sigma_scale=0.2, steps=5):
                """通过多步扩散过程向图像添加高斯噪声"""
                noisy_image = image.copy()
                sigma_base = np.std(image) * sigma_scale

                for step in range(steps):
                    mean = 0
                    # 每一步逐渐增加噪声的标准差
                    sigma = sigma_base * (step + 1)
                    gauss = np.random.normal(mean, sigma, image.shape)
                    noisy_image = noisy_image + gauss

                    # 确保图像数值仍然在合理范围内（例如，对于0-255的像素值）
                    noisy_image = np.clip(noisy_image, 0, 255)

                return noisy_image.astype(np.uint8)


            def calculate_distance(img1, img2):
                """计算两幅图像之间的欧氏距离"""
                return np.linalg.norm(img1 - img2)

            def load_image(path):
                """从给定路径加载图像，并转换为灰度"""
                image = Image.open(path).convert('RGB')
                return np.array(image)

            # 处理所有sampled_items并加载图像
            items_with_images = [(item, load_image(item.impath)) for item in sampled_items]

            # 计算每个图像的不熟悉度
            distances = []
            for item, image in items_with_images:
                distance_sum = 0
                for _ in range(num_samples):
                    noised_image = gaussian_noise(image)
                    distance_sum += calculate_distance(image, noised_image)
                average_distance = distance_sum / num_samples
                distances.append((item, average_distance))

            # 根据平均距离排序并选择最大的shots个
            selected_items = sorted(distances, key=lambda x: x[1], reverse=True)[:shots]

            # 返回一个新的sampled_items列表，只包含选出的样本
            return [item for item, distance in selected_items]

        def ensure_directory_exists(folder):
            """确保指定的文件夹存在，如果不存在，则创建它"""
            if not os.path.exists(folder):
                os.makedirs(folder)
            return folder

        def save_data_with_pickle(data, num_shots, dataset_name):
            folder = ensure_directory_exists('montecarlo_selection_by_class')
            filename = os.path.join(folder,
                                    f"{num_shots}_{dataset_name}.pkl")  # Incorporate num_shots into the filename
            with open(filename, 'wb') as file:
                pickle.dump(data, file)
            print(f"Data saved to {filename}")

        def load_data_with_pickle(num_shots, dataset_name):
            folder = 'montecarlo_selection_by_class'
            filename = os.path.join(folder,
                                    f"{num_shots}_{dataset_name}.pkl")  # Incorporate num_shots into the filename
            if os.path.exists(filename):
                with open(filename, 'rb') as file:
                    data = pickle.load(file)
                return data
            else:
                return None


        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot noise_dataset')

        output = load_data_with_pickle(num_shots,dataset_name)  # 尝试加载整个数据集

        if output is None:
            print(f'montecarlo_selection_by_class')
            output = []
            for data_source in data_sources:
                tracker = self.split_dataset_by_label(data_source)
                dataset = []

                for label, items in tracker.items():
                    if len(items) >= 2 * num_shots:
                        # print(len(items))
                        sampled_items = random.sample(items, 2 * num_shots)
                        sampled_items = montecarlo_selection_by_class(sampled_items,num_samples = 3,shots = num_shots)

                    else:
                        if repeat:
                            sampled_items = random.choices(items, k= 2 * num_shots)
                            sampled_items = montecarlo_selection_by_class(sampled_items, num_samples=3, shots=num_shots)
                        else:
                            sampled_items = items

                    dataset.extend(sampled_items)

            output.append(dataset)

            save_data_with_pickle(output, num_shots,dataset_name)

        if len(output) == 1:
            return output[0]

        return output


    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.domain].append(item)

        return output


class DatasetWrapper(TorchDataset):
    def __init__(self, data_source, input_size, transform=None, is_train=False,
                 return_img0=False, k_tfm=1):
        self.data_source = data_source
        self.transform = transform # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = k_tfm if is_train else 1
        self.return_img0 = return_img0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                'Cannot augment the image {} times '
                'because transform is None'.format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = T.InterpolationMode.BICUBIC
        to_tensor = []
        to_tensor += [T.Resize(input_size, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        normalize = T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        )
        to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            'label': item.label,
            'domain': item.domain,
            'impath': item.impath
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = 'img'
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output['img'] = img

        if self.return_img0:
            output['img0'] = self.to_tensor(img0)

        return output['img'], output['label']

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


def build_data_loader(
    data_source=None,
    batch_size=64,
    input_size=224,
    tfm=None,
    is_train=True,
    shuffle=False,
    dataset_wrapper=None
):

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(data_source, input_size=input_size, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        num_workers=8,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=(torch.cuda.is_available())
    )
    assert len(data_loader) > 0

    return data_loader
