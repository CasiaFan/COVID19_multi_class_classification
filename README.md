Here is the tutorial to train a 3-class COVID-19 classification model.

### 0. Requirements
- Python 3.7
- PyTorch 
- TorchVision
- Fire
- OpenCV 4.2.0
- Numpy
- Pandas
- Scikit-Learn
- Pillow

### 1. Data
- Chest X-ray images: make the COVIDx dataset following this tutorial: https://github.com/lindawangg/COVID-Net/blob/master/create_COVIDx.ipynb
**train_COVIDx5.txt** and **test_COVIDx4.txt**: train and test files of COVIDx dataset
**train_sample.txt**: toy sample for testing the code.

- CT images: make the COVIDx-CT dataset as: https://github.com/haydengunraj/COVIDNet-CT/blob/master/create_COVIDx_CT.ipynb

### 2. Code
**Currently, only CXR datasets are supported (3-classes)**. 
`data.py`: prepare data for training. 
`model.py`: network architectures that could be employed for training
`train.py`: code to train the model (**Read it and understand the basic steps to train a network using PyTorch**)
`eval.py`: evaluate trained model performance: F1 score, precision, recall, sensitivity, AUC  

Train: `bash run_train.sh`, or `python train.py --help` for help to check how to set the parameters
Test: `bash run_eval.sh`, or `python eval.py --help` for help. 

### 3. Tasks
1. add `dataset` function for CT images as shown in `data.py`
2. **adjust the training process in `train.py` to support multi-modal training**
3. support evaluation of multi-modal models in `eval.py`
4. try to add model network architectures in `model.py`
   
### 4. Reference Repos: 
1. https://github.com/lindawangg/COVID-Net
2. https://github.com/haydengunraj/COVIDNet-CT