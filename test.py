import cv2
import torch
import numpy as np
import utils
from constant import PROJECT_PATH, MEAN, STD
from train import Train
from table_dataset import TableDataset


class Test(Train):
    def __init__(self, model_name, weight_epoch_name, img_width=512, img_height=128, device='cuda:0'):
        super().__init__(model_name, device=device, weight_epoch_name=weight_epoch_name,
                         img_width=img_width, img_height=img_height, data_path=None)
        self.model.eval()

    def single_test(self, img, draw_img=True):
        origin = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.erode(img, np.ones([3, 3], dtype=np.uint8), iterations=5)
        img, ratio_w, ratio_h = TableDataset.standard_resize(img, self.img_height, self.img_width)
        img = (img - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
        img = img.transpose([2, 0, 1])
        img = np.expand_dims(img, 0)
        img = torch.tensor(img).float()
        img, pad_w, pad_h = utils.pad(img, 512, 128)
        img = img.to(self.device)

        with torch.no_grad():
            mask = self.model(img)

            if self.model_name in {'unet', 'ma-unet', 'test'}:
                mask = mask
            elif self.model_name in {'unet++', 'unet3+'}:
                mask = mask[-1] if self.model_name == 'unet++' else mask[0]

            mask = mask.cpu().detach().numpy().squeeze()
            mask[mask > 0.5] = 255
            mask = mask.astype(np.uint8)

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cells = []
            for i, contour in enumerate(contours):
                rect = cv2.boundingRect(contour)
                cell = [rect[0], rect[1], int(rect[0] + rect[2]), int(rect[1] + rect[3])]
                cell[0] = round((cell[0] - pad_w) / ratio_w)
                cell[1] = round((cell[1] - pad_h) / ratio_h)
                cell[2] = round((cell[2] - pad_w) / ratio_w)
                cell[3] = round((cell[3] - pad_h) / ratio_h)
                cells.append(cell)
                if draw_img:
                    start_p = (cell[0], cell[1])
                    end_p = (cell[2], cell[3])
                    random_color = (np.random.rand(3) * 255)
                    random_color = [int(color) for color in random_color]
                    cv2.rectangle(origin, start_p, end_p, random_color, 6)

        return origin, cells


if __name__ == '__main__':
    test = Test(model_name='ma-unet', device='cuda:3', weight_epoch_name='latest')
    img = cv2.imread(f'{PROJECT_PATH}/img/origin.png')
    img, cells = test.single_test(img)
