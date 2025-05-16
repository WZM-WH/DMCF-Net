import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from data_loader import ToTensorLab
from data_loader import FloodDataset, RandomFourSplitFlipAndRecombine

from model import DMCFNet
from loss import Focalloss

FLloss = Focalloss.FocalLoss(reduction='mean')

def log_to_file(log_path, message):
    with open(log_path, 'a') as log_file:
        log_file.write(message + '\n')
if __name__ == '__main__':
    # ------- 1. define loss function --------
    def LossFunction(tensor, labels_v):
        loss2 = FLloss(tensor, labels_v)
        return loss2
    # ------- 2. set the directory of training dataset --------
    model_name = 'DMCF-Net'

    data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    tra_image_dir = os.path.join('images' + os.sep)
    tra_label_dir = os.path.join('labels' + os.sep)
    image_ext = '.tif'
    label_ext = '.tif'
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
    os.makedirs(model_dir, exist_ok=True)

    log_file_path = os.path.join(model_dir, f"{model_name}_training_log.txt")
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    epoch_num = 300
    batch_size_train = 3
    batch_size_val = 1
    train_num = 0
    val_num = 0

    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]
        imidx = ".".join(img_name.split(".")[:-1])
        tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")
    print("---")

    train_num = len(tra_img_name_list)

    val_image_dir = os.path.join('val_images' + os.sep)
    val_label_dir = os.path.join('val_labels' + os.sep)
    val_img_name_list = glob.glob(data_dir + val_image_dir + '*' + image_ext)
    val_lbl_name_list = []
    for img_path in val_img_name_list:
        img_name = img_path.split(os.sep)[-1]
        imidx = ".".join(img_name.split(".")[:-1])
        val_lbl_name_list.append(data_dir + val_label_dir + imidx + label_ext)
    val_num = len(val_img_name_list)
    salobj_dataset = FloodDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([ToTensorLab(), RandomFourSplitFlipAndRecombine()]))
    salobj_dataset_val = FloodDataset(
        img_name_list=val_img_name_list,
        lbl_name_list=val_lbl_name_list,
        transform=transforms.Compose([ToTensorLab()]))

    salobj_dataloader_train = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0,
                                         drop_last=True, pin_memory=True)
    salobj_dataloader_val = DataLoader(salobj_dataset_val, batch_size=batch_size_val, shuffle=False, num_workers=0,
                                       drop_last=False, pin_memory=True)

    # ------- 3. define model --------
    net = DMCFNet.DMCFNet(in_channels=2, out_channels=1)

    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    ite_num4val = 0
    save_frq = 100

    best_iou = 0
    best_f1 = 0
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    train_loss_list = []
    val_loss_list = []
    iou_list = []
    f1_list = []

    for epoch in range(0, epoch_num):
        net.train()
        running_loss = 0.0
        ite_num4val = 0

        for i, data in enumerate(salobj_dataloader_train):
            ite_num += 1
            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            optimizer.zero_grad()
            output = net(inputs_v)
            loss = LossFunction(output, labels_v)

            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
                running_loss += loss.data.item()
                ite_num4val += 1
            else:
                print(f"[epoch: {epoch + 1}, batch: {i + 1}] Loss is NaN or Inf. Skipping this batch.")

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f" % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val))

        net.eval()

        val_loss = 0.0
        total_iou = 0.0
        total_f1 = 0.0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        ite = 0
        with torch.no_grad():
            for data_val in salobj_dataloader_val:
                inputs_val, labels_val = data_val['image'], data_val['label']
                inputs_val = inputs_val.type(torch.FloatTensor)
                labels_val = labels_val.type(torch.FloatTensor)

                if torch.cuda.is_available():
                    inputs_val_v, labels_val_v = Variable(inputs_val.cuda(), requires_grad=False), Variable(
                        labels_val.cuda(), requires_grad=False)
                else:
                    inputs_val_v, labels_val_v = Variable(inputs_val, requires_grad=False), Variable(labels_val, requires_grad=False)

                output_val = net(inputs_val_v)
                loss_val = LossFunction(output_val, labels_val_v)

                if not torch.isnan(loss_val):
                    val_loss += loss_val.item()

                    ite += 1

                    pred = torch.sigmoid(output_val)
                    pred = (pred > 0.5).cpu().numpy()
                    pred = pred.astype(np.int32)
                    target = labels_val_v.cpu().numpy()
                    valid_mask = target != -1
                    pred = pred[valid_mask]
                    target = target[valid_mask]

                    tp = np.sum((pred == 1) & (target == 1))
                    fp = np.sum((pred == 1) & (target == 0))
                    fn = np.sum((pred == 0) & (target == 1))

                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

        final_iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
        final_f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

        train_loss_list.append(running_loss / ite_num4val)
        val_loss_list.append(val_loss / ite)
        iou_list.append(final_iou)
        f1_list.append(final_f1)

        scheduler.step()
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}: Current learning rate: {current_lr}")
        print(f"Epoch {epoch + 1}, Valid IoU: {final_iou:.6f}, Valid F1 Score: {final_f1:.6f}")
        print(f"Epoch {epoch + 1}, Valid Recall: {recall:.6f}, Valid Precision: {precision:.6f}")
        print("[epoch: %d/%d] validation loss: %f, train loss: %f" % (epoch + 1, epoch_num, val_loss / ite, running_loss / ite_num4val))

        avg_val_loss = val_loss / ite
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = (f"[{current_time}] Epoch {epoch + 1}: Train Loss: {running_loss / ite_num4val:.6f}, "
                       f"Val Loss: {avg_val_loss:.6f}, "
               f"IoU: {final_iou:.6f}, F1: {final_f1:.6f}, "
               f"Recall: {recall:.6f}, Precision: {precision:.6f}, LR: {current_lr:.8f}")
        log_to_file(log_file_path, log_message)

        if (epoch + 1) % save_frq == 0:
            torch.save(net.state_dict(), model_dir + model_name + "_%depoch_train_%3f_val_%3f.pth" % (epoch + 1, running_loss / ite_num4val, val_loss / ite))

        Valloss = val_loss / ite
        Trainloss = running_loss / ite_num4val

        if Valloss < best_val_loss and Trainloss < best_train_loss:
            best_val_loss = Valloss
            best_train_loss = Trainloss
            save_path = os.path.join(model_dir, f"{model_name}_Lossbest.pth")
            torch.save(net.state_dict(), save_path)

        if final_iou > best_iou and final_f1 > best_f1:
            best_iou = final_iou
            best_f1 = final_f1
            save_path = os.path.join(model_dir, f"{model_name}_IouF1_best.pth")
            torch.save(net.state_dict(), save_path)

        if final_f1 > 0.8:
            save_path = os.path.join(model_dir, f"{model_name}_F1_above_0.8_epoch_{epoch + 1}.pth")
            torch.save(net.state_dict(), save_path)
            print(f"Model saved as F1 > 0.8 at {save_path}")

        running_loss = 0.0

plt.figure(figsize=(10, 6))
plt.plot(range(1, epoch_num + 1), train_loss_list, label='Train Loss', linewidth=1)
plt.plot(range(1, epoch_num + 1), val_loss_list, label='Validation Loss', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
figure_path = os.path.join(model_dir, 'loss_curve.png')
plt.savefig(figure_path)
plt.show()
plt.figure(figsize=(10, 6))

plt.plot(range(1, epoch_num + 1), iou_list, label='IoU', linewidth=1)
plt.plot(range(1, epoch_num + 1), f1_list, label='F1 Score', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.title('IoU, F1, Recall, and Precision Over Epochs')
plt.legend()
plt.grid()
metrics_figure_path = os.path.join(model_dir, 'metrics_curve.png')
plt.savefig(metrics_figure_path)
plt.show()