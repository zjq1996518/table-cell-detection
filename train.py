import torch
from tqdm import tqdm

from model import UNet, MAUNet, NestedUNet, UNet3Plus
import table_dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

from utils import calc_target


class Train(object):
    def __init__(self, model_name, data_path, img_width=512, img_height=128, lr=1e-4, weight_decay=1e-5, epoch=100, batch_size=4, device='cuda:0',
                 weight_epoch_name=None):

        self.device = device
        self.epoch = epoch
        self.batch_size = batch_size
        self.model_name = model_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = self.load_model(model_name, weight_epoch_name)

        self.img_width = img_width
        self.img_height = img_height

        self.data_path = data_path
        self.train_dataset, self.val_dataset = None, None
        self.train_data_loader, self.val_data_loader = None, None

    def load_model(self, model_name='unet', weight_epoch_name=None):
        model = None
        if model_name == 'unet':
            model = UNet()
        elif model_name == 'unet++':
            model = NestedUNet()
        elif model_name == 'ma-unet':
            model = MAUNet()
        elif model_name == 'unet3+':
            model = UNet3Plus()
        if weight_epoch_name is not None:
            model.load_weight(weight_epoch_name)
        model.to(self.device)
        return model

    def start(self):
        self.train_dataset, self.val_dataset = table_dataset.allocation_dataset(self.data_path, self.img_width, self.img_height)
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        exp_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
        for e in range(self.epoch):
            with tqdm(total=len(self.train_data_loader)) as bar:
                self.model.train()
                avg_loss = 0
                avg_acc = 0
                avg_recall = 0
                avg_f1 = 0
                for i, (x, label) in enumerate(self.train_data_loader):
                    x = x.to(self.device)
                    label = label.to(self.device)
                    label = label.unsqueeze(1)
                    optimizer.zero_grad()
                    loss = 0
                    if self.model_name in {'unet', 'ma-unet'}:
                        y = self.model(x)
                        loss = F.binary_cross_entropy_with_logits(y, label)
                    # unet++ unet3+ ??????????????????????????????loss?????????????????????
                    elif self.model_name in {'unet++', 'unet3+'}:
                        ys = self.model(x)
                        for y in ys:
                            loss += F.binary_cross_entropy_with_logits(y, label)
                        loss /= len(ys)
                        y = ys[-1] if self.model_name == 'unet++' else ys[0]

                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item()
                    # ????????????0.5
                    y[y >= 0.5] = 1.
                    y[y < 0.5] = 0.
                    with torch.no_grad():
                        # ????????????????????????????????? f1??????
                        acc, recall, f1 = calc_target(y, label)
                        avg_acc += acc
                        avg_recall += recall
                        avg_f1 += f1
                        bar.set_postfix({
                            'loss': avg_loss / (i + 1),
                            'acc': avg_acc / (i + 1),
                            'recall': avg_recall / (i + 1),
                            'f1': avg_f1 / (i + 1)
                        })
                        bar.update(1)

            exp_lr.step()
            self.eval_model(e+1)

    def eval_model(self, epoch):
        with tqdm(total=len(self.val_data_loader)) as bar:
            self.model.eval()
            avg_acc = 0
            avg_recall = 0
            avg_f1 = 0
            with torch.no_grad():
                for i, (x, label) in enumerate(self.val_data_loader):
                    x = x.to(self.device)
                    label = label.to(self.device)
                    label = label.unsqueeze(1)
                    if self.model_name in {'unet', 'ma-unet'}:
                        y = self.model(x)
                    elif self.model_name in {'unet++', 'unet3+'}:
                        ys = self.model(x)
                        y = ys[-1] if self.model_name == 'unet++' else ys[0]

                    # ????????????0.5
                    y[y >= 0.5] = 1.
                    y[y < 0.5] = 0.
                    with torch.no_grad():
                        # ????????????????????????????????? f1??????
                        acc, recall, f1 = calc_target(y, label)
                        avg_acc += acc
                        avg_recall += recall
                        avg_f1 += f1
                        bar.set_postfix({
                            'acc': avg_acc / (i + 1),
                            'recall': avg_recall / (i + 1),
                            'f1': avg_f1 / (i + 1)
                        })
                        bar.update(1)

        self.model.save_weight(epoch=epoch, f1=avg_f1 / (i + 1))


if __name__ == '__main__':
    train = Train('unet', data_path='/local/aitrain/zjq/unet-data', device='cuda:2')
    train.start()
