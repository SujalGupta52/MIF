import sys
import time
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
from utils import my_scheduler, AvgACC, save_model , build_cache_model , pre_load_features
from eval import Eval

    
def clip_forward(clip_model, images, text_feats):
    image_feats, mlp_logits, new_feats = clip_model.encode_image(images.cuda())
    image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
    clip_logits = 100. * image_feats @ text_feats
    return image_feats, clip_logits, mlp_logits, new_feats

class Trainer:
    
    def __init__(self, cfg, clip_model, train_loader, test_loader, logger, text_feats, cache_keys, cache_values, val_features, val_labels, test_features, test_labels,val_loader=None ):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.clip_model = clip_model
        self.train_loader = train_loader
        self.cache_keys = cache_keys
        self.cache_values = cache_values
        self.val_features = val_features
        self.bal_labels = val_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.logger = logger
        self.checkpoint_dir = cfg['checkpoint_dir']
        self.save_interval = cfg['save_interval']
        self.epochs = cfg['train_epoch']
        self.log_dir = f"./log/{self.checkpoint_dir.split('/')[-1]}"


        self.adapter_catch = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
        self.adapter_catch.weight = nn.Parameter(self.cache_keys.t())
        self.optimizer = torch.optim.AdamW(list(self.clip_model.parameters()) + list(self.adapter_catch.parameters()), lr=cfg['lr'], eps=1e-4)
        self.scheduler = my_scheduler(self.optimizer, cfg['lr'], 1e-6, self.epochs, len(self.train_loader), 10)

        self.text_feats = text_feats
        self.eval = Eval(self.cfg, self.clip_model, test_loader, self.text_feats, self.logger)
        self.Zeta, self.epsilon , self.beta = self.cfg['init_Zeta'], self.cfg['init_epsilon'] ,self.cfg['init_beta']


    def get_loss(self, labels, clip_logits, mlp_logits, feats, new_feats ,cache_logits):


        l1_loss = F.l1_loss(mlp_logits, clip_logits)

        tip_logits = clip_logits + self.epsilon * cache_logits

        tip_loss = F.cross_entropy(tip_logits, labels)

        ce_loss = F.cross_entropy(mlp_logits, labels)

        gammar = self.cfg['gammar']

        loss = gammar * l1_loss + ce_loss + tip_loss
        return loss, [l1_loss, ce_loss,tip_loss]




    def train_mode(self):
        self.adapter_catch.train()


    def train_epoch(self, epoch):
        self.train_mode()
        train_loss = 0.0
        ACC = AvgACC()

        loss_list = [0, 0, 0]

        self.adapter_catch.train()


        with tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Training epoch {epoch}") as tqdm_train:

            for _, (images, labels) in tqdm_train:

                images, labels = images.cuda(), labels.cuda()

                feats, clip_logits, mlp_logits, new_feats = clip_forward(self.clip_model, images, self.text_feats)
                with torch.no_grad():
                    image_features = feats

                affinity = self.adapter_catch(image_features)

                cache_logits = (self.Zeta * (affinity - 1)) @ self.cache_values
                loss, losses = self.get_loss(labels, clip_logits, mlp_logits, feats, new_feats ,cache_logits )

                if torch.isnan(loss):
                    self.logger.info(f"{self.cfg['desc']}:!!! Loss is NaN. Program terminated.")
                    sys.exit()


                logits =  clip_logits + self.epsilon * cache_logits  +  self.beta * mlp_logits
                ACC.step(logits, labels)
                train_loss += loss.item()
                for i, l in enumerate(losses):
                    loss_list[i] += l.item()
                tqdm_train.set_postfix(cur_loss=loss.item())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
            train_acc = ACC.cal()
            train_loss = train_loss / len(self.train_loader)
            
        print(f"{epoch=}, {loss_list=}")
        if epoch == self.epochs - 1:

            self.logger.info(f"[l1_loss,ce_loss,tip_loss] => {loss_list}")


        return train_acc * 100, train_loss
        
    def train(self):
        self.logger.info('-------------------- START TRAINING --------------------')
        train_name = self.logger.name
        train_st = time.time()
        # self.validate()
        
        for epoch in range(self.epochs):
            epoch_st = time.time()
            self.logger.info(f'====> Epoch: {epoch}')
            train_acc, train_loss = self.train_epoch(epoch)
            epoch_ed = time.time()
            self.logger.info(f"      train_acc: {train_acc:.4f} %    train_loss: {train_loss:.4f}    train_time: {(epoch_ed - epoch_st):.4f} s    lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:.8f}")
            # self.validate()
            # if (epoch % cfg['save_interval'] == 0 and epoch != 0):
                # save_model(self.checkpoint_dir, f'{train_name}_epoch_{epoch}.pth', self.clip_model)
                # self.validate()
            if epoch == self.epochs - 1:
                # save_model(self.checkpoint_dir, f'{train_name}_last.pth', self.clip_model)
                self.validate()
        
        duration = int(time.time() - train_st)
        self.logger.info(f'Total time used for training: {duration // 3600} h {duration % 3600 // 60} min {duration % 60} sec')
        
    def validate(self):
        test_features = self.test_features
        val_features = self.val_features
        affinity_test = self.adapter_catch(test_features)
    
        affinity_val = self.adapter_catch(val_features)
        

        cache_values = self.cache_values

        self.eval.clip_model = self.clip_model
        self.adapter_catch.eval()

        seach_hp = None
        best_beta, best_Zeta, best_epsilon = None, None, None

        if self.val_loader:
            self.eval.val_loader = self.val_loader
            best_Zeta, best_epsilon ,best_beta= self.eval.eval(affinity_val,cache_values)
            seach_hp = True

        if self.cfg['dataset'] != 'imagenet':
            self.eval.val_loader = self.test_loader
            self.eval.eval(affinity_test,cache_values,best_beta, best_Zeta, best_epsilon,seach_hp = seach_hp)
