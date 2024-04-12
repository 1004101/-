import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model_backbone import mobile_repvgg_small as create_model
from utils import read_split_data, train_one_epoch, evaluate, read_split_data_mode


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    train_images_path, train_images_label = read_split_data_mode(args.data_path, mode='train', supported=['.JPEG'])
    val_images_path, val_images_label = read_split_data_mode(args.data_path, mode='val', supported=['.JPEG'])

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        # # 删除有关分类类别的权重
        # for k in list(weights_dict.keys()):
        #     if "classifier" in k:
        #         del weights_dict[k]

        # 删除有关分类类别和 layer4 和 layer5 的权重
        for k in list(weights_dict.keys()):
            if "layer_1" in k:
                del weights_dict[k]
            if "layer_2" in k:
                del weights_dict[k]
        model.load_state_dict(weights_dict, strict=False)

    # if args.freeze_layers:
    #     for name, para in model.named_parameters():
    #         # 除head外，其他权重全部冻结
    #         if "classifier" not in name:
    #             para.requires_grad_(False)
    #         else:
    #             print("training {}".format(name))


    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1E-2)

    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/best_model.pth")

        torch.save(model.state_dict(), "./weights/latest_model.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0002)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,
    #                     default="E:/迅雷下载/处理好的ImageNet2012/imagenet")
    parser.add_argument('--data-path', type=str,
                        default="/root/autodl-tmp/imagenet")

    # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weights', type=str, default='F:/deep-learning-for-image-processing-master/pytorch_classification/MobileViT/weights/latest_model.pth',
    #                     help='initial weights path')
    parser.add_argument('--weights', type=str,
                        default='/root/autodl-tmp/MobileViT/weights/best_model.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)