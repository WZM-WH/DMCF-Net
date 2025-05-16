import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import glob
import cv2
from data_loader import ToTensorLab
from data_loader import FloodDataset
from model import DMCFNet


def main():
    # --------- 1. get image path and name ---------
    model_file_name = 'DMCFNet'
    model_name = 'DMCFNet_Lossbest'

    image_dir = os.path.join(os.getcwd(), 'test_data', 'images')
    label_dir = os.path.join(os.getcwd(), 'test_data', 'labels')
    prediction_dir = os.path.join(os.getcwd(), 'results', 'DMCFNet' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_file_name, model_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    label_name_list = glob.glob(label_dir + os.sep + '*')

    test_salobj_dataset = FloodDataset(img_name_list=img_name_list,
                                       lbl_name_list=label_name_list,
                                       transform=transforms.Compose([ToTensorLab()])
                                       )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=0)

    # --------- 3. model define ---------
    net =  DMCFNet.DMCFNet(in_channels=2, out_channels=1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    for i_test, data_test in enumerate(test_salobj_dataloader):
        print("Predicting:", img_name_list[i_test].split(os.sep)[-1])
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        output = net(inputs_test)

        pred = output[:, 0, :, :]
        pred = F.sigmoid(pred)

        pred_binary = (pred >= 0.5).float()

        pred_np = pred_binary.cpu().detach().numpy()
        pred_np = (pred_np * 255).astype(np.uint8)

        pred_img = pred_np[0]
        mask_name_png = img_name_list[i_test].split(os.sep)[-1]
        mask_name_png = mask_name_png.split('.')[0] + '.png'

        save_path = os.path.join(prediction_dir, mask_name_png)
        cv2.imwrite(save_path, pred_img)

if __name__ == "__main__":
    main()
