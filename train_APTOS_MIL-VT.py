import copy
import time
from datetime import datetime, timedelta

import PIL
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import os

import utils
from data_manager.dataset import dataset_Aptos
from loss.MultiClassMetrics import *
from models.FinetuneVTmodels import MIL_VT_FineTune
from utils import *
from models.MIL_VT import *
from torchvision.transforms import InterpolationMode

####################################

def main():

    """Basic Setting"""
    data_path = "/content/diabetic_retinopathy_dataset/colored_images"  # ✅ Flattened image directory
    csv_path = "/content/"                  # ✅ Path where CSV is saved
    save_model_path = '/content/drive/MyDrive/MIL_VT/PytorchModel/'  # Save models to Google Drive
    csvName = "/content/diabetic_retinopathy_dataset/trainLabels.csv"

    gpu_ids = [0]
    start_epoch = 0
    max_epoch = 30
    save_fraq = 10

    batch_size = 16
    img_size = 384
    initialLR = 2e-5
    n_classes = 5

    balanceFlag = True  #balanceFlag is set to True to balance the sampling of different classes
    debugFlag = False  #Debug flag is set to True to train on a small dataset

    base_model = 'MIL_VT_small_patch16_'+str(img_size)  #nominate the MIL-VT model to be used
    MODEL_PATH_finetune = '/content/MIL_VT_weights/fundus_pretrained_VT_small_patch16_384_5Class.pth.tar'

    dateTag = datetime.today().strftime('%Y%m%d')
    prefix = base_model + '_' + dateTag
    model_save_dir = os.path.join(save_model_path, prefix)  # Save models in Google Drive
    tbFileName = os.path.join(model_save_dir, 'runs/' + prefix)
    savemodelPrefix = prefix + '_ep'

    # Create directories for logs and checkpoints if they don't exist
    os.makedirs(os.path.join(save_model_path, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(save_model_path, 'checkpoints'), exist_ok=True)

    ##resume training with an interrupted model
    resumeFlag = False
    resumeEpoch = 0
    resumeModel = 'path to resume model'

    print('####################################################')
    print('Save model Path', model_save_dir)
    print('Save training record Path', tbFileName)
    print('####################################################')

    #################################################
    sys.stdout = Logger(os.path.join(model_save_dir,
                     savemodelPrefix[:-3] + 'log_train-%s.txt' % time.strftime("%Y-%m-%d-%H-%M-%S")))
    tbWriter = SummaryWriter(tbFileName)

    torch.cuda.set_device(gpu_ids[0])
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.get_device_name(gpu_ids[0]), torch.cuda.get_device_capability(gpu_ids[0]))

    #################################################
    """Set up the model, loss function and optimizer"""

    ## set the model and assign the corresponding pretrain weight
    model = MIL_VT_FineTune(base_model, MODEL_PATH_finetune, num_classes=n_classes)
    model = model.cuda()
    if len(gpu_ids) >= 2:
        model = DataParallel(model, device_ids=gpu_ids)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    multiLayers = list()
    for name, layer in model._modules.items():
        if name.__contains__('MIL_'):
            multiLayers.append({'params': layer.parameters(), 'lr': 5*initialLR})
        else:
            multiLayers.append({'params': layer.parameters()})
    optimizer = torch.optim.Adam(multiLayers, lr = initialLR, eps=1e-8, weight_decay=1e-5)

    from timm.scheduler import CosineLRScheduler

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=max_epoch,
        lr_min=1e-6,
        warmup_lr_init=1e-6,
        warmup_t=3,
    )


    from torch.nn import CrossEntropyLoss

    # Adjust weights based on class distribution
    weights = torch.tensor([1.0, 2.5, 2.0, 3.0, 3.0]).cuda()  # tweak these for your dataset
    criterion = CrossEntropyLoss(weight=weights)


    if resumeFlag:
        print(" Loading checkpoint from epoch '%s'" % (
             resumeEpoch))
        checkpoint = torch.load(resumeModel)
        initialLR = checkpoint['lr']
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print('Model weight loaded')

    #################################################
    """Load the CSV as DF and split train / valid set"""
    DF0 = pd.read_csv(csvName, encoding='UTF')
    DF0 = DF0.rename(columns={'level': 'diagnosis'}) 

    if debugFlag == True:
        indexes = np.arange(len(DF0))
        np.random.seed(0)
        np.random.shuffle(indexes)
        DF0 = DF0.iloc[indexes[:600], :]
        DF0 = DF0.reset_index(drop=True)

    indexes = np.arange(len(DF0))
    np.random.seed(0)
    np.random.shuffle(indexes)
    trainNum = int(len(indexes)*0.7)
    valNum = int(len(indexes)*0.8)
    DF_train = DF0.loc[indexes[:trainNum]]
    DF_val = DF0.loc[indexes[trainNum:valNum]]
    DF_test = DF0.loc[indexes[valNum:]]
    DF_train = DF_train.reset_index(drop=True)
    DF_val = DF_val.reset_index(drop=True)
    DF_test = DF_test.reset_index(drop=True)

    print('Train: ', len(DF_train), 'Val: ', len(DF_val), 'Test: ', len(DF_test))
    for tempLabel in [0,1,2,3,4]:
        print(tempLabel, np.sum(DF_train['diagnosis']==tempLabel),\
                        np.sum(DF_val['diagnosis']==tempLabel),
                        np.sum(DF_test['diagnosis']==tempLabel))

    #################################################

    from torchvision.transforms import RandAugment, InterpolationMode
    import torchvision.transforms as transforms
    
    
    transform_train = transforms.Compose([
        transforms.Resize((img_size + 40, img_size + 40)),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(hue=0.05, saturation=0.05, brightness=0.05),
        RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    
    
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])



    dataset_train = dataset_Aptos(data_path, DF_train, transform = transform_train)
    dataset_valid = dataset_Aptos(data_path, DF_val, transform = transform_test)
    dataset_test = dataset_Aptos(data_path, DF_test, transform=transform_test)

    """assign sample weight to deal with the unblanced classes"""
    weights = make_weights_for_balanced_classes(DF_train, n_classes)                                                           
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    if balanceFlag == True:
        train_loader = DataLoader(dataset_train, batch_size,
                              sampler = sampler,
                              num_workers=2,  drop_last=True, shuffle=False) #shuffle=False when using the balance sampler,
    else:
        train_loader = DataLoader(dataset_train, batch_size, num_workers=2,  drop_last=True, shuffle=True) #shuffle=True,
    valid_loader = DataLoader(dataset_valid, batch_size, num_workers=2, drop_last=False)
    test_loader = DataLoader(dataset_test, batch_size, num_workers=2,  drop_last=False)

    #################################################

    """The training procedure"""

    start_time = time.time()
    train_time = 0
    best_perform = 0
    for epoch in range(start_epoch, max_epoch + 1):
        start_train_time = time.time()
        currentLR = 0

        for param_group in optimizer.param_groups:
            currentLR = param_group['lr']
        print('lr:', currentLR)

        train(epoch, model, criterion, optimizer, train_loader, max_epoch, tbWriter)
        train_time += np.round(time.time() - start_train_time)

        AUC_val, wF1_val = \
            val(epoch, model, criterion, valid_loader, max_epoch, tbWriter)

        if wF1_val > best_perform and debugFlag == False:
            best_perform = wF1_val
            state_dict = model_without_ddp.state_dict()
            saveCheckPointName = savemodelPrefix + '_bestmodel.pth.tar'
            utils.save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
            }, os.path.join(model_save_dir, saveCheckPointName))
            best_model = copy.deepcopy(model)
            print('Checkpoint saved, ', saveCheckPointName)

        if epoch>0 and (epoch) % save_fraq == 0 and debugFlag == False:

            state_dict = model_without_ddp.state_dict()
            wF1_val = round(wF1_val, 3)
            saveCheckPointName = savemodelPrefix + str(epoch) + '_' + str(wF1_val) + '.pth.tar'
            utils.save_checkpoint({
                'state_dict': state_dict,
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
            }, os.path.join(model_save_dir, saveCheckPointName))
            print('Checkpoint saved, ', saveCheckPointName)


        scheduler.step(epoch)
    
    elapsed = np.round(time.time() - start_time)
    elapsed = str(timedelta(seconds=elapsed))
    train_time = str(timedelta(seconds=train_time))


    print('###################################################')
    print('Performance on Test Set with last model')
    test(epoch, model, criterion, test_loader, tbWriter)

    if debugFlag == False and 'best_model' in locals():
        print('Performance on Test Set with best model')
        test(epoch, best_model, criterion, test_loader, tbWriter)
    else:
        print('Skipping test with best_model (not saved in debug mode)')

    tbWriter.close()
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
