import cv2
import torch
import numpy as np
import utils
from train import Train
from table_dataset import TableDataset


class Test(Train):
    def __init__(self, model_name, weight_epoch_name, img_width=512, img_height=128, device='cuda:0'):
        super().__init__(model_name, device=device, weight_epoch_name=weight_epoch_name,
                         img_width=img_width, img_height=img_height, data_path=None)
        self.model.eval()

    def single_test(self, img, draw_img=True):
        img = cv2.erode(img, np.ones([3, 3], dtype=np.uint8), iterations=5)
        img = TableDataset.standard_resize(img, self.img_height, self.img_width)
        origin = img.copy()

        img = img.transpose([2, 0, 1])
        img = np.expand_dims(img, 0)
        img = torch.tensor(img).float().cuda(3)
        img = img / 255
        img = utils.tensor_pad32m(img, 1)

        with torch.no_grad():
            mask = self.model(img)

            if self.model_name in {'unet', 'ma-unet', 'test'}:
                mask = mask
            elif self.model_name in {'unet++', 'unet3+'}:
                mask = mask[-1] if self.model_name == 'unet++' else mask[0]

            mask = mask * 255

            mask = mask.cpu().detach().numpy().squeeze()
            mask = mask.astype(np.uint8)
            mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cells = []

            for i, contour in enumerate(contours):
                rect = cv2.boundingRect(contour)
                cells.append([rect[0], rect[1], rect[2], rect[3]])

                if draw_img:
                    start_p = (rect[0], rect[1])
                    end_p = (int(rect[0] + rect[2]), int(rect[1] + rect[3]))
                    random_color = (np.random.rand(3) * 255)
                    random_color = [int(color) for color in random_color]
                    cv2.rectangle(origin, start_p, end_p, random_color, 6)

        if draw_img:
            return origin, cells

        return cells


if __name__ == '__main__':
    test = Test(model_name='unet++', device='cuda:3', weight_epoch_name='5')
    img = np.random.rand(512, 1024, 3).astype(np.uint8)
    img, cells = test.single_test(img)
    cv2.imshow('test', img)
    cv2.waitKey(0)
    print(cells)

