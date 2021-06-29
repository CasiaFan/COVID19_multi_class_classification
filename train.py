import torch
import torch.optim as optim 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data import prepare_data, image_sampler
from model import C_net
import copy 
import time, os
import fire
import numpy as np

CUR_DIR = os.path.dirname(os.path.realpath(__file__))

def train(model, 
          model_save_path, 
          dataloader, 
          datasize,
          optimizer, 
          criterion, 
          num_epochs, 
          device="cpu"):
    """Train Classifier"""
    best_acc = 0
    train_hist = {}
    train_hist["C_loss"] = []
    train_hist["acc"] = []
    start_t = time.time() 
    # add log to tensorborad 
    # writer = SummaryWriter(log_dir=model_save_path+"/event")
    # writer.add_graph(model, (torch.rand(1, 3, input_size, input_size).to(device), ))
    best_test_model = os.path.join(model_save_path, "best_model.pt") 
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-" * 50)
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0
            running_corrects = 0
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # if use_cent_loss:
                #     optimizer_cent.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs[0], labels)
                    # print(outputs[0], labels)
                    _, preds = torch.max(outputs[0], 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            ds = datasize[phase]
            epoch_loss = running_loss / ds 
            epoch_acc = running_corrects.double().item() / ds 
            # writer.add_scalar("Loss/train", epoch_loss)
            # writer.add_scalar("Acc/train", epoch_acc)
            print("{} Loss: {:.4f}, Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            if phase == "train":
                train_hist["C_loss"].append(epoch_loss)
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_test_model)
            if phase == "test":
                train_hist["acc"].append(epoch_acc)
        if not ((epoch+1) % 5):
            torch.save(model.state_dict(), model_save_path+"/w_epoch_{}.pt".format(epoch+1))
    time_elapsed = time.time() - start_t 
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    print("Best val acc: {:.4f}".format(best_acc))
    # model.load_state_dict(torch.load(best_model_w, map_location=torch.device(device)))
    # writer.close()
    return model, train_hist

def run(model_name, 
        image_dir, 
        image_size=224, 
        num_classes=3, 
        batch_size=16, 
        num_epochs=40, 
        model_save_path="train_res", 
        device="cuda:0", 
        lr=0.001, 
        moment=0.9, 
        use_pretrained=True,
        loss="cross-entropy",
        dataset="covidx",
        balanced_sampling=None, 
        num_samples=None):
    """
    loss: loss type, including cross-entropy, focal, hem 
    gamma: gamma factor for focal loss
    ratio: topk ratio of loss to compute for HEM 
    balanced_sampling: whether to balance the number of each classes during training
    num_samples: used for balanced sampling. Total number of samples to augment. If None, average to the mean number. 
    """
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    # get classifier 
    C = C_net(model_name=model_name,
              num_classes=num_classes, 
              use_pretrained=use_pretrained, return_logit=True).to(device)
    if dataset == "covidx":
        train_file = os.path.join(CUR_DIR, "train_COVIDx5.txt")
        test_file = os.path.join(CUR_DIR, "test_COVIDx5.txt")
    elif dataset == "test":
        train_file = os.path.join(CUR_DIR, "toy_sample.txt")
        test_file = os.path.join(CUR_DIR, "toy_sample.txt")
        dataset = "covidx"
    config = {"image_size": image_size, "train": train_file, "test": test_file, "dataset": dataset}
    image_datasets, datasizes = prepare_data(image_dir, config)
    dataloaders = {}
    for phase in ["train", "test"]:
        sampler=image_sampler(image_datasets[phase], training=phase=="train", balanced_sampling=balanced_sampling, num_samples=num_samples)
        shuffle = phase=="train"
        if sampler:
            shuffle = False
            datasizes[phase] = sampler.num_samples
        dataloaders[phase] = torch.utils.data.DataLoader(image_datasets[phase], shuffle=shuffle, batch_size=batch_size, num_workers=0, drop_last=True, sampler=sampler)
    # loss function
    if sampler:
        cls_weight = [1.0, 1.0, 1.0]
    else:
        cls_weight = [1.0, 4.0, 1.0]
    if loss == "cross-entropy":
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(cls_weight)).to(device)
    # optimizer
    print("optimized parameter names")
    for name, param in C.named_parameters():
        if param.requires_grad == True:
            print("\t", name)
    print("-"*50)
    optimizer = optim.SGD(C.parameters(), lr=lr, momentum=moment) 
    model_fit, hist = train(model=C, 
                           model_save_path=model_save_path, 
                           dataloader=dataloaders, 
                           datasize=datasizes,
                           optimizer=optimizer, 
                           criterion=criterion, 
                           num_epochs=num_epochs, 
                           device=device)
    # torch.save(model_ft.state_dict(), model_save_path+'/best_model.pt') 
    for k, v in hist.items():
        print("{}: {}".format(k, v))

if __name__ == "__main__":
    fire.Fire(run)
    # # training config
    # input_size = 224
    # num_classes = 3 
    # batch_size = 16
    # num_epoches = 40
    # model_name = "resnet50"
    # device = "cuda:0"
    # input_dir = "/shared/anastasio5/COVID19/data/covidx"
    # model_save_path = "covidx_res50"

