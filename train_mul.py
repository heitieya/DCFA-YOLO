import datetime
import os
from functools import partial
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.yolo_mul import YoloBody
from nets.yolo_training import (Loss, ModelEMA, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks_mul import EvalCallback, LossHistory
from utils.dataloader_mul import YoloDataset, yolo_dataset_collate
from utils.utils import (download_weights, get_classes, seed_everything,
                         show_config, worker_init_fn)
from utils.utils_fit_mul import fit_one_epoch

if __name__ == "__main__":

    Cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # New parameter: custom filename for saving results
    custom_filename = "test"  # This can be modified as needed

    # Seed for random number generation
    seed = 11

    # Flag for distributed training
    distributed = False

    # Flag for synchronized batch normalization
    sync_bn = False

    # Flag for using mixed precision training
    fp16 = False

    # Path to the classes file
    classes_path = 'model_data/voc_classes.txt'

    # Path to the model weights
    model_path = ''

    # Input shape for the model
    input_shape = [640, 640]

    # Model size parameter
    phi = 's'

    # Flag for using pretrained weights
    pretrained = False

    # Flag for using mosaic data augmentation
    mosaic = False
    mosaic_prob = 0.5  # Probability of applying mosaic augmentation
    mixup = False
    mixup_prob = 0.5  # Probability of applying mixup augmentation
    special_aug_ratio = 0.7  # Ratio for special augmentation

    # Label smoothing parameter
    label_smoothing = 0

    # Initial epoch for training
    Init_Epoch = 0
    # Epochs to freeze training
    Freeze_Epoch = 0
    # Batch size during freeze training
    Freeze_batch_size = 0

    # Total epochs for unfreezing training
    UnFreeze_Epoch = 200
    # Batch size after unfreezing
    Unfreeze_batch_size = 16

    # Flag for freezing training
    Freeze_Train = False

    # Initial learning rate
    Init_lr = 1e-2
    # Minimum learning rate
    Min_lr = Init_lr * 0.01

    # Type of optimizer to use
    optimizer_type = "sgd"
    # Momentum for the optimizer
    momentum = 0.937
    # Weight decay for regularization
    weight_decay = 5e-4

    # Learning rate decay type
    lr_decay_type = "cos"

    # Period for saving the model
    save_period = 20

    # Directory to save logs
    save_dir = 'logs'

    # Flag for evaluation during training
    eval_flag = True
    # Period for evaluation
    eval_period = 20

    # Number of workers for data loading
    num_workers = 4

    # Paths for training and validation annotations
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    seed_everything(seed)


    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0


    class_names, num_classes = get_classes(classes_path)

    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi)
            dist.barrier()
        else:
            download_weights(phi)

    model = YoloBody(input_shape, num_classes, phi, pretrained=pretrained)


    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k.startswith('backbone.'):
                k_rgb = k.replace('backbone.', 'backbone_rgb.')
                k_nir = k.replace('backbone.', 'backbone_nir.')
                if k_rgb in model_dict and np.shape(model_dict[k_rgb]) == np.shape(v):
                    temp_dict[k_rgb] = v
                    load_key.append(k_rgb)
                if k_nir in model_dict and np.shape(model_dict[k_nir]) == np.shape(v):
                    temp_dict[k_nir] = v
                    load_key.append(k_nir)
            elif k in model_dict and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")


    yolo_loss = Loss(model)


    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, f"{custom_filename}_{time_str}")
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not supported in single GPU mode.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()


    ema = ModelEMA(model_train)


    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path=classes_path, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size,
            Freeze_Train=Freeze_Train, Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type,
            momentum=momentum, lr_decay_type=lr_decay_type, save_period=save_period, save_dir=save_dir,
            num_workers=num_workers, num_train=num_train, num_val=num_val
        )

    UnFreeze_flag = False

    if Freeze_Train:
        for param in model.backbone_rgb.parameters():
            param.requires_grad = False
        for param in model.backbone_nir.parameters():
            param.requires_grad = False

    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size


    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    optimizer = {
        'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
        'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
    }[optimizer_type]
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2})

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)


    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")


    if ema:
        ema.updates = epoch_step * Init_Epoch


    train_dataset = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True,
                                special_aug_ratio=special_aug_ratio)
    val_dataset = YoloDataset(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                              mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)


    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
        batch_size = batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler,
                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

    if local_rank == 0:
        eval_callback = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                     eval_flag=eval_flag, period=eval_period)
    else:
        eval_callback = None


    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size

            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

            for param in model.backbone_rgb.parameters():
                param.requires_grad = True
            for param in model.backbone_nir.parameters():
                param.requires_grad = True

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            if ema:
                ema.updates = epoch_step * epoch

            if distributed:
                batch_size = batch_size // ngpus_per_node

            gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
            gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler,
                                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

            UnFreeze_flag = True

        gen.dataset.epoch_now = epoch
        gen_val.dataset.epoch_now = epoch

        if distributed:
            train_sampler.set_epoch(epoch)


        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)


        fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                      epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir,
                      local_rank)

        if distributed:
            dist.barrier()

    if local_rank == 0:
        loss_history.writer.close()
