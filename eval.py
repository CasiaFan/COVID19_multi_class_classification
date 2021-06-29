import torch 
from model import C_net
from data import prepare_data
import fire
import numpy as np
from scipy import stats
from collections import OrderedDict
from PIL import Image 
from torchvision import transforms
import os
from sklearn.metrics import roc_auc_score, auc, roc_curve
# from sklearn.preprocessing import label_binarize

CUR_DIR = os.path.dirname(os.path.realpath(__file__))

COVID_LABELS = ["normal", "COVID-19", "pneumonia"]

def vote(x):
    res = {}
    for i in x:
        if i in x:
            res[x] += 1
        else:
            res[x] = 1
    return max(res)

def read_image_tensor(image_path, image_size):
    # read image
    img = Image.open(image_path)
    img = img.convert("RGB")
    # resize 
    img = img.resize((image_size, image_size))
    # normalize
    img = np.asarray(img)/127.5 - 1 
    img = np.expand_dims(img, 0)
    # to tensor
    img = torch.tensor(img).type(torch.float32)
    # NHWC to NCHW
    img = img.permute(0, 3, 1, 2)
    return img

def calculate_auc(pred, label, num_classes=3):
    fpr, tpr = {}, {}
    # label = label_binarize(label, classes=list(range(num_classes)))
    label = np.eye(num_classes)[label]
    if num_classes > 1:
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(label[:, i], pred[:,i])
    else:
        fpr[0], tpr[0], _ = roc_curve(label, pred)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):    
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr = mean_tpr / num_classes
    auc_value = auc(all_fpr, mean_tpr)
    return auc_value, all_fpr, mean_tpr

def predict(model_name, 
            num_classes, 
            model_weights, 
            image_dir=None, 
            image_size=224,
            image_path=None, 
            device="cpu",  
            dataset="covidx",
            is_unet=False,
            multi_gpus=True, 
            auc=False):
    """
    model_name: classifier name
    num_classes: number of classes
    model_weights: model weights path
    image_dir: testing image directory
    image_path: single image path for evaluation
    dataset: name of testing dataset
    is_unet: whether testing model is unet
    multi_gpus: whether model is trained on multi-gpus
    """
    if device == "cpu":
        gpu_ids = []
    else:
        gpu_ids = [0]
    if is_unet:
        model = define_G(input_nc=3, output_nc=3, ngf=64, netG=model_name, gpu_ids=gpu_ids)
    else:
        model = C_net(model_name=model_name, num_classes=num_classes, use_pretrained=False, return_logit=True).to(device)
    state_dict=torch.load(model_weights, map_location=torch.device(device))
    if multi_gpus:
        new_state_dict = OrderedDict()
        # remove 'module.' of dataparallel
        for k, v in state_dict.items():
            name = k[7:] 
            new_state_dict[name]=v
    else:
        new_state_dict = state_dict
    model.load_state_dict(new_state_dict)
    model.eval()
    if image_path:
        img = read_image_tensor(image_path, image_size)
        outputs = model(img)
        if is_unet:
            _, pred = torch.max(outputs[1], 1)
        else:
            _, pred = torch.max(outputs[0], 1)
        return pred
    else:
        if dataset == "covidx":
            train_file = os.path.join(CUR_DIR, "dataset/train_COVIDx5.txt")
            test_file = os.path.join(CUR_DIR, "dataset/test_COVIDx5.txt")
        elif dataset == "HN":
            train_file = "/shared/radon/TOP/train_HN_sample.txt"
            test_file = "/shared/radon/TOP/test_HN_sample.txt"
        elif dataset == "test": 
            train_file = os.path.join(CUR_DIR, "example/train_sample.txt")
            test_file = os.path.join(CUR_DIR, "example/train_sample.txt")
            dataset = "covidx"
        config = {"image_size": image_size, "train": train_file, "test": test_file, "dataset": dataset}
        image_datasets, datasizes = prepare_data(image_dir, config)
        dataloader = torch.utils.data.DataLoader(image_datasets["test"], shuffle=False)  

        # result matrics:
        #      __________________________________________
        #      | gt \ pred | Normal | COVID | Pneumonia | 
        #      ------------------------------------------
        #      |  Normal   |        |       |           |
        #      |  COVID    |        |       |           |
        #      |  Pneumonia|        |       |           |
        #      ------------------------------------------
        if dataset == "covidx":
            result_matrics = np.zeros((3, 3))
        elif dataset == "HN":
            result_matrics = np.zeros((2, 2)) 
        with torch.no_grad():
            if auc:
                pred_list, label_list = None, []
            for img, label in dataloader:
                tag = label.numpy()[0]
                img = img.to(device)
                label = label.to(device)
                outputs = model(img)
                if is_unet:
                    _, pred = torch.max(outputs[1], 1)
                    score = outputs[1].numpy()
                else:
                    _, pred = torch.max(outputs[0], 1)
                    score = outputs[0].numpy()
                if auc:
                    if pred_list is None:
                        pred_list = score
                    else:
                        pred_list = np.concatenate([pred_list, score], axis=0)
                    label_list.append(tag)
                pred = int(pred.item())
                result_matrics[tag][pred] += 1
            # comput auc 
            if auc:
                auc_v, fpr, tpr = calculate_auc(pred_list, label_list, num_classes=num_classes)
                print("AUC: ", auc_v)
                print("FPR: ", fpr)
                print("TRP", tpr)
        return result_matrics


def mean_confidence_interval(x, confidence=0.95):
    # get CI with 0.95 confidence following normal gaussian distribution
    n = len(x)
    m, se = np.mean(x), stats.sem(x)
    ci = stats.t.ppf((1 + confidence) / 2., n-1) * se
    # ci = 1.96 * se  # assume gaussian distribution
    return m, ci

def eval(model_name, 
         num_classes, 
         model_weights, 
         image_dir, 
         image_size=224, 
         device="cpu",
         dataset="covidx",
         is_unet=False,
         multi_gpus=True, 
         auc=False):
    """
    Output metrics of model, including: Precision, Sensitivity, AUC
    """
    result_matrics = predict(model_name, 
                            num_classes, 
                            model_weights, 
                            image_dir, 
                            image_size=image_size, 
                            device=device, 
                            dataset=dataset,
                            is_unet=is_unet,
                            multi_gpus=multi_gpus, 
                            auc=auc)
    # precision: TP / (TP + FP)
    print("result matrics: ", result_matrics)
    # res_acc = [result_matrics[i, i]/np.sum(result_matrics[:,i]) for i in range(num_classes)]
    res_acc = []
    # sensitivity: TP / (TP + FN)
    res_sens = []
    # res_sens = [result_matrics[i, i]/np.sum(result_matrics[i,:]) for i in range(num_classes)]
    # specificity: TN / (TN+FP)
    res_speci = []
    # f1 score: 2TP/(2TP+FP+FN)
    f1_score = []
    for i in range(num_classes):
        TP = result_matrics[i,i]
        FN = np.sum(result_matrics[i,:])-TP
        spe_matrics = np.delete(result_matrics, i, 0)
        FP = np.sum(spe_matrics[:, i])
        TN = np.sum(spe_matrics) - FP
        acc = TP/(TP+FP)
        sens = TP/(TP+FN)
        speci = TN/(TN+FP)
        f1 = 2*TP/(2*TP+FP+FN)
        res_acc.append(acc)
        res_speci.append(speci)
        res_sens.append(sens)
        f1_score.append(f1)
    if dataset == "covidx":
        print('Precision: Normal: {0:.3f}, COVID: {1:.3f}, Pneumonia: {2:.3f}, avg: {3:.3f}'.format(res_acc[0],res_acc[1],res_acc[2], np.mean(res_acc)))
        print('Sensitivity: Normal: {0:.3f}, COVID: {1:.3f}, Pneumonia: {2:.3f}, avg: {3:.3f}'.format(res_sens[0],res_sens[1],res_sens[2], np.mean(res_sens)))
        print('Specificity: Normal: {0:.3f}, COVID: {1:.3f}, Pneumonia: {2:.3f}, avg: {3:.3f}'.format(res_speci[0],res_speci[1],res_speci[2], np.mean(res_speci)))
        print('F1 score: Normal: {0:.3f}, COVID: {1:.3f}, Pneumonia: {2:.3f}, avg: {3:.3f}'.format(f1_score[0],f1_score[1],f1_score[2], np.mean(f1_score)))          
    elif dataset == 'HN':
        print('Precision: w/o: {0:.3f}, with: {1:.3f}, avg: {2:.3f}'.format(res_acc[0],res_acc[1], np.mean(res_acc)))
        print('Sensitivity: w/o: {0:.3f}, with: {1:.3f}, avg: {2:.3f}'.format(res_sens[0], res_sens[1], np.mean(res_sens)))
        print('Specificity: w/o: {0:.3f}, with: {1:.3f}, avg: {2:.3f}'.format(res_speci[0],res_speci[1], np.mean(res_speci)))
        print('F1 score: w/o: {0:.3f}, with: {1:.3f}, avg{2:.3f}'.format(f1_score[0],f1_score[1], np.mean(f1_score))) 
    else:
        print("unknown dataset")


def example_ci():
    res_m = np.array([
[89.2, 	100., 	93.0 ,	94.0, 	99.0, 	89.0, 	93.0, 	93.6, 	94.0, 	100.0, 	96.5, 	96.8, 	93.8, 	94.2, 	93.0, 	93.7], 
[89.2, 	96.7, 	90.8 ,	92.2, 	99.0, 	88.0, 	89.0, 	92.0, 	94.0, 	98.5 ,	95.5 ,	96.0 ,	93.8 ,	92.1 ,	90.0 ,	92.0 ],
[91.4, 	96.8, 	91.1 ,	93.1, 	96.0, 	91.0, 	92.0, 	93.0, 	95.5, 	98.5 ,	95.5 ,	96.5 ,	93.7 ,	93.8 ,	91.5 ,	93.0 ],
[91.6, 	96.7, 	90.1 ,	92.8, 	98.0, 	89.0, 	91.0, 	92.7, 	95.5, 	98.5 ,	95.0 ,	96.3 ,	94.7 ,	92.7 ,	90.5 ,	92.6 ],
[88.0, 	94.7, 	94.8 ,	92.5, 	95.0, 	90.0, 	92.0, 	92.3, 	93.5, 	97.5 ,	97.5 ,	96.2 ,	91.3 ,	92.3 ,	93.4 ,	92.4 ],
[93.3, 	91.2, 	93.6 ,	92.7, 	97.0, 	93.0, 	88.0, 	92.7, 	96.5, 	95.5 ,	97.0 ,	96.3 ,	95.1 ,	92.1 ,	90.7 ,	92.6 ],
[90.9,	95.1,   92.4,	92.8,	96.5,   90.3,	91.2	,92.6,	95.1,	97.6,	96.3,	96.3,	93.6,	92.6,	91.8,	92.7]

    ])
    for i in range(res_m.shape[1]):
        m, ci = mean_confidence_interval(res_m[:, i])
        print("mean: {}; CI: {}".format(m, ci))


if __name__ == "__main__":
    # fire.Fire(eval)
    
    eval("x", 3, "x", "x", dataset="covidx")
    # example_ci()
    
    # test restore image
    # image = "/shared/anastasio5/COVID19/data/covidx/train/covid-19-pneumonia-14-PA.png"
    # cls_w = "/shared/anastasio5/COVID19/covid19_classification_with_gan/covidx_train/gan_res50_224_patch_ssim/best_model.pt"
    # gan_w = "/shared/anastasio5/COVID19/covid19_classification_with_gan/covidx_train/gan_res50_224_patch_ssim/G_epoch_180.pt"
    # image_size = 224
    # cnet = "resnet50"
    # gnet = "info"
    # label_code = [0, 1, 0]
    # bottom_width = 7
    # init_conv_channels = 128
    # with_cls = False
    # save_file = "rec_covid.jpg"
    # reconstruct(image_path=image, label_code=label_code, c_name=cnet, c_weight=cls_w, g_name=gnet, g_weight=gan_w, image_size=image_size, device="cpu", save_file=save_file, embedding_dim=128, bottom_width=bottom_width, init_conv_channels=init_conv_channels, with_cls=with_cls)
    
    # # test auc
    # from scipy.special import softmax
    # x1 = np.random.random((20, 3))
    # x2 = np.random.random((20, 3))
    # label1 = [np.random.randint(3) for i in range(20)]
    # label2 = [np.random.randint(3) for i in range(20)]
    # prob1 = softmax(x1, axis=1)
    # prob2 = softmax(x2, axis=1)
    # auc1, fpr1, tpr1 = calculate_auc(prob1, label1, num_classes=3)
    # auc2, fpr2, tpr2 = calculate_auc(prob2, label2, num_classes=3)
    # data = {"res50": {"auc": auc1, "fpr": fpr1, "tpr": tpr1},
    # "res18": {"auc": auc2, "fpr": fpr2, "tpr": tpr2}}
    # draw_auc(data, "test/auc.png")
    
