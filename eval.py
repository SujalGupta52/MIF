import torch
from tqdm import tqdm

from utils import AvgACC, cal_acc



def fuse_logits(mlp_logits, clip_logits, cache_logits,  epsilon ,beta):
    logits = clip_logits + epsilon * cache_logits  +  beta * mlp_logits

    return logits



class Eval:
    
    def __init__(self, cfg, clip_model, val_loader, text_feats, logger) -> None:
        self.cfg = cfg
        self.clip_model = clip_model
        self.text_feats = text_feats
        self.val_loader = val_loader
        self.logger = logger
        self.batch_size = cfg['batch_size']
        
    def evaluate_epoch(self, images):    
        image_feats, mlp_logits, _ = self.clip_model.encode_image(images)
        image_feats /= image_feats.norm(dim=-1, keepdim=True)
        clip_logits = 100. * image_feats @ self.text_feats
        return clip_logits, mlp_logits

    def eval(self, affinity,cache_values,best_beta= None, best_Zeta= None, best_epsilon= None,seach_hp = None):
        ACC = AvgACC()
        self.clip_model.eval()
        all_clip_logits = []
        all_mlp_logits = []
        all_labels = []
        self.best_epsilon = best_epsilon
        self.best_beta = best_beta
        self.best_Zeta = best_Zeta

        with torch.no_grad():
            with tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc='Evaluate') as tqdm_eval:
                for _, (images, labels) in tqdm_eval:
                    clip_logits, mlp_logits = self.evaluate_epoch(images.cuda())
                    ACC.step(mlp_logits, labels)
                    all_clip_logits.append(clip_logits)
                    all_mlp_logits.append(mlp_logits)
                    all_labels.append(labels)
        
        self.all_clip_logits = torch.cat(all_clip_logits, dim=0)
        self.all_mlp_logits = torch.cat(all_mlp_logits, dim=0)
        self.all_labels = torch.cat(all_labels, dim=0)




        if seach_hp:
            print('test data classfication begin')

            cache_logits = self.best_Zeta * (affinity - 1) @ cache_values

            logits = fuse_logits(self.all_mlp_logits , self.all_clip_logits , cache_logits, self.best_epsilon , self.best_beta)



            acc = cal_acc(logits, self.all_labels) * 100.
            print(f"test best acc: {acc:}")

            self.logger.info(f"{self.cfg['desc']} :*** test best beta Zeta epsilon =  {best_Zeta:.4f} {best_epsilon:.4f}=> {acc:.2f}% [{acc=}]")
            return

        best_Zeta, best_epsilon, best_beta,best_acc = self.search_hp(self.cfg, affinity, cache_values, self.all_labels,
                                                                      self.all_clip_logits, self.all_mlp_logits)

        print(f"val best acc: {best_acc:}")
        self.logger.info(
            f"{self.cfg['desc']} :*** val best acc =   {best_acc:.2f}% [{best_acc=}]")

        return  best_Zeta, best_epsilon ,best_beta


    def search_hp(self, cfg, affinity, cache_values, labels, clip_logits, mlp_logits):
        def generate_search_list(best_value, scale, step, fine=False, allow_negative=False):
            if fine:
                # 精细搜索时仍保持正值，或根据需要调整
                start = max(best_value - scale / 2, 0)  # 避免生成负值
                return [start + step * i for i in range(5)]
            else:
                start = -scale if allow_negative else 0.1  # 添加负值范围控制
                end = scale
                return [start + i * (end - start) / step for i in range(step + 1)]

        if cfg['search_hp']:
            # 第1阶段：粗搜索，减少搜索步长以覆盖大范围
            beta_list = generate_search_list(0, cfg['search_scale'][0], 10)
            Zeta_list = generate_search_list(0, cfg['search_scale'][1], 10)
            epsilon_list = generate_search_list(0, cfg['search_scale'][2], 10, allow_negative=False)  # 允许负数

            best_acc = 0
            best_Zeta, best_epsilon, best_beta = 0, 0, 0

            # 粗搜索
            for beta in beta_list:
                for Zeta in Zeta_list:
                    cache_logits = (Zeta * (affinity - 1)) @ cache_values
                    for epsilon in epsilon_list:
                        logits = clip_logits + epsilon * cache_logits + beta * mlp_logits

                        acc = cal_acc(logits, labels) * 100
                        if acc > best_acc:
                            print(
                                f"New best setting in coarse search, Zeta: {Zeta:.2f}, epsilon: {epsilon:.2f}, beta: {beta:.2f}; accuracy: {acc:.2f}")
                            best_acc = acc
                            best_Zeta = Zeta
                            best_epsilon = epsilon
                            best_beta = beta

            print(
                f"\nAfter coarse search, best accuracy: {best_acc:.2f}, Zeta: {best_Zeta:.2f}, epsilon: {best_epsilon:.2f}, beta: {best_beta:.2f}.\n")

            # 第2阶段：精细搜索
            beta_list = generate_search_list(best_beta, 1, 0.1, fine=True)
            Zeta_list = generate_search_list(best_Zeta, 1, 0.1, fine=True)
            epsilon_list = generate_search_list(best_epsilon, 1, 0.1, fine=True)  # 精细搜索阶段默认不需要负数

            for beta in beta_list:
                for Zeta in Zeta_list:
                    cache_logits = (Zeta * (affinity - 1)) @ cache_values
                    for epsilon in epsilon_list:
                        logits = clip_logits + epsilon * cache_logits + beta * mlp_logits

                        acc = cal_acc(logits, labels) * 100
                        if acc > best_acc:
                            print(
                                f"New best setting in fine search, Zeta: {Zeta:.2f}, epsilon: {epsilon:.2f}, beta: {beta:.2f}; accuracy: {acc:.2f}")
                            best_acc = acc
                            best_Zeta = Zeta
                            best_epsilon = epsilon
                            best_beta = beta

            print(
                f"\nAfter fine search, best accuracy: {best_acc:.2f}, Zeta: {best_Zeta:.2f}, epsilon: {best_epsilon:.2f}, beta: {best_beta:.2f}.\n")

        return best_Zeta, best_epsilon, best_beta, best_acc






