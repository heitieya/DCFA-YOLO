import os

import torch
from tqdm import tqdm

from utils.utils import get_lr

def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    # Initialize loss variables
    loss        = 0
    val_loss    = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        # Get RGB and NIR images
        images_rgb, images_nir, bboxes = batch
        with torch.no_grad():
            if cuda:
                images_rgb = images_rgb.cuda(local_rank)
                images_nir = images_nir.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)

        # Zero the gradients
        optimizer.zero_grad()

        if not fp16:
            # Forward pass, handling both RGB and NIR modalities
            outputs = model_train(images_rgb, images_nir)
            loss_value = yolo_loss(outputs, bboxes)

            # Backward pass
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # Forward pass, handling both RGB and NIR modalities
                outputs = model_train(images_rgb, images_nir)
                loss_value = yolo_loss(outputs, bboxes)

            # Backward pass
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()

        if ema:
            ema.update(model_train)

        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1), 'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        # Get RGB and NIR images
        images_rgb, images_nir, bboxes = batch
        with torch.no_grad():
            if cuda:
                images_rgb = images_rgb.cuda(local_rank)
                images_nir = images_nir.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass, handling both RGB and NIR modalities
            outputs = model_train_eval(images_rgb, images_nir)
            loss_value = yolo_loss(outputs, bboxes)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print(f'Epoch: {epoch + 1}/{Epoch}')
        print(f'Total Loss: {loss / epoch_step:.3f} || Val Loss: {val_loss / epoch_step_val:.3f}')

        # Save weights
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(loss_history.log_dir, f"ep{epoch + 1:03d}-loss{loss / epoch_step:.3f}-val_loss{val_loss / epoch_step_val:.3f}.pth"))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(loss_history.log_dir, "best_epoch_weights.pth"))

        torch.save(save_state_dict, os.path.join(loss_history.log_dir, "last_epoch_weights.pth"))
