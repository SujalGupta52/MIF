import os
import torch
import clip
import json
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm


def setup_seed(seed):
    if seed == 0:
        print('random seed')
        torch.backends.cudnn.benchmark = True
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)


def get_clip_feat_dim(clip_model, img=torch.ones((1, 3, 224, 224))):
    clip_model.eval()
    with torch.no_grad():
        output = clip_model.encode_image(img.cuda())
        print(f"{output.shape=}")
    return output.shape[1]


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as fp:
        return json.load(fp)


def log_init(logger, cfg):
    logger.info('**************************************************************')
    logger.info(f'Here are the args:')
    for arg in cfg.keys():
        logger.info(f'{arg} : {cfg[arg]}')


def make_dirs(*kargs):
    for dir in kargs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def make_dirs_from_file(*kargs):
    dirs = []
    for path in kargs:
        dirs.append(os.path.split(path)[0])
    make_dirs(*dirs)


def get_model_param_size(model):
    size = sum(param.numel() for param in model.parameters())
    return size


def save_model(save_dir, name, model, optimizer=None, epoch=0, lr_scheduler=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None
    }
    torch.save(checkpoint, os.path.join(save_dir, name))


def load_model(checkpoint_path, model, 
               optimizer=None, lr_scheduler=None, key_filter=lambda key: True):
    ckp = torch.load(checkpoint_path, map_location='cpu')
    net_state_dict = ckp['net']
    model_state_dict = model.state_dict()
    net_state_dict = {k: v for k, v in net_state_dict.items() if k in model_state_dict and key_filter(k)}
    model_state_dict.update(net_state_dict)
    model.load_state_dict(model_state_dict)
    if optimizer and ckp['optimizer']:
        optimizer.load_state_dict(ckp['optimizer'])
    if lr_scheduler and ckp['lr_scheduler']:
        lr_scheduler.load_state_dict(ckp['lr_scheduler'])
    return model.cuda(), optimizer, lr_scheduler, ckp['epoch']


def cal_acc_mid(logits, labels):
    pred = torch.argmax(logits, -1)
    acc_num = (pred == labels.cuda()).sum().item()
    total = len(labels)
    return acc_num, total

def cal_acc(logits, labels):
    acc_num, total = cal_acc_mid(logits, labels)
    acc = 1.0 * acc_num / total
    return acc


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc
class AvgACC:
    def __init__(self) -> None:
        self.acc_num = 0
        self.total = 0
    
    def step(self, logits, labels):
        acc_num, total = cal_acc_mid(logits, labels)
        self.acc_num += acc_num
        self.total += total
    
    def cal(self):
        return 0.00 if self.total == 0 else 1.0 * self.acc_num / self.total


class my_scheduler:
    
    def __init__(self, optimizer, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0) -> None:
        self.optimizer = optimizer
        self.optimizer.param_groups[0]['lr'] = 0
        
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(0, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        self.schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        self.schedule = np.concatenate((warmup_schedule, self.schedule))
        self.id = 0
        assert len(self.schedule) == epochs * niter_per_ep
    
    def step(self):
        self.optimizer.param_groups[0]['lr'] = self.schedule[self.id]
        self.id += 1


# the following function is modified from Tip-Adapter

def clip_classifier(feat_path, classnames, template, clip_model):
    if os.path.exists(feat_path):
        print(f"Loading texture features from {feat_path}")
        text_feats = torch.load(feat_path, map_location='cpu')
        return text_feats.cuda()
    
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            if isinstance(template, list):
                texts = [t.format(classname) for t in template]
            elif isinstance(template, dict):
                texts = template[classname]
                
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
        
        make_dirs_from_file(feat_path)
        torch.save(clip_weights, feat_path)
            
    return clip_weights

def gpt_clip_classifier(classnames, gpt_prompts, clip_model, template):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):
    if cfg['load_cache'] == False:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))

                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features, mlp_logits, new_feats = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)

                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
        safe_backbone = cfg['backbone'].replace('/', '_')

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots" + '_' + safe_backbone + '_' + cfg['dataset'] +".pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots" + '_' +  safe_backbone +  '_' + cfg['dataset'] +".pt")

    else:
        safe_backbone = cfg['backbone'].replace('/', '_')
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots" + '_' + safe_backbone + '_'+ cfg['dataset'] +".pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots" + '_' + safe_backbone + '_'+ cfg['dataset'] +".pt")

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):
    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features, mlp_logits, new_feats = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)
        safe_backbone = cfg['backbone'].replace('/', '_')
        torch.save(features, cfg['cache_dir'] + "/"  + cfg['dataset'] + '_' + str(cfg['shots']) + "shots" + safe_backbone+  '_'+ split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + cfg['dataset'] + '_' + str(cfg['shots']) + "shots" + safe_backbone+ '_' + split + "_l.pt")

    else:
        safe_backbone = cfg['backbone'].replace('/', '_')
        features = torch.load(cfg['cache_dir'] + "/"  + cfg['dataset'] + '_' + str(cfg['shots']) + "shots" + safe_backbone+ '_' + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + cfg['dataset'] + '_' + str(cfg['shots'])+ "shots" + safe_backbone+ '_' + split + "_l.pt")

    return features, labels


def logits_fuse(zero_logtis, logits, normalize='mean'):
    # normalize logits
    softmax_fun = nn.Softmax(dim=1)
    if normalize == 'softmax':
        zero_logtis = softmax_fun(zero_logtis)
    elif normalize =='linear':
        zero_logtis /= torch.norm(zero_logtis, p=2, dim=1, keepdim=True)
    elif normalize == 'mean':
        logits_std = torch.std(zero_logtis, dim=1, keepdim=True)
        logits_mean = torch.mean(zero_logtis, dim=1, keepdim=True)
        zero_logtis = (zero_logtis - logits_mean) / logits_std
    else:
        raise("error normalize!")
    similarity_matrix = []
    normalize_logits = []
    for logit in logits:
        if normalize == 'softmax':
            current_normalize_logits = softmax_fun(logit)
        elif normalize =='linear':
            current_normalize_logits = logit / torch.norm(logit, p=2, dim=1, keepdim=True)
        elif normalize == 'mean':
            logits_std = torch.std(logit, dim=1, keepdim=True)
            logits_mean = torch.mean(logit, dim=1, keepdim=True)
            current_normalize_logits = (logit - logits_mean) / logits_std
        else:
            raise("error normalize!")
        current_similarity = current_normalize_logits * zero_logtis
        current_similarity = torch.sum(current_similarity, dim=1, keepdim=True)
        similarity_matrix.append(current_similarity)
        normalize_logits.append(current_normalize_logits)
    similarity_matrix = torch.stack(similarity_matrix, dim=-2)
    similarity_matrix = softmax_fun(similarity_matrix)
    normalize_logits = torch.stack(normalize_logits, dim=-2)
    result_logits = torch.sum(normalize_logits * similarity_matrix, dim=1)

    return result_logits
