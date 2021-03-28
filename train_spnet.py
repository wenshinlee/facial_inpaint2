import time
import tqdm
import torch
import torchvision

import core.utils as utils
from datasets.dataset import get_dataloader
from core.SPNet.spnet_trainer import SegInpaintModel


def get_train_parameter(hyperparameters):
    epoch_count = hyperparameters['epoch_count']
    niter = hyperparameters['niter']
    niter_decay = hyperparameters['niter_decay']
    batch_size = hyperparameters['batch_size']
    display_freq = hyperparameters['image_save_iter']
    print_freq = hyperparameters['log_iter']
    save_epoch_freq = hyperparameters['snapshot_save_iter']
    checkpoints_dir = hyperparameters['checkpoints_dir']
    model_name = hyperparameters['model_name']
    return epoch_count, niter, niter_decay, batch_size, display_freq, print_freq, \
           save_epoch_freq, checkpoints_dir, model_name


if __name__ == '__main__':
    hy_config = utils.get_config('core/SPNet/configs/celeba-hq.yaml')
    data_loader = get_dataloader(hyperparameters=hy_config)
    model = SegInpaintModel(hy_config)

    epoch_count, niter, niter_decay, batch_size, display_freq, \
    print_freq, save_epoch_freq, checkpoints_dir, model_name = get_train_parameter(hy_config)

    visuals = utils.Visuals(checkpoints_dir, model_name)

    total_steps = 0
    for epoch in range(epoch_count, niter + niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for n_iter, data in tqdm.tqdm(enumerate(data_loader, 0), total=len(data_loader),
                                      desc="epoch-->%d" % epoch, ncols=80, leave=False):
            iter_start_time = time.time()
            total_steps += batch_size
            epoch_iter += batch_size
            model.set_input(data)
            model.optimize_parameters()
            #
            if epoch_iter % print_freq == 0:
                errors = model.get_current_errors()
                t = round((time.time() - iter_start_time) / batch_size, 4)
                # discriminator
                visuals.add_scalar('discriminator/loss_D_fake', errors['loss_D_fake'], total_steps + 1)
                visuals.add_scalar('discriminator/loss_D_real', errors['loss_D_real'], total_steps + 1)
                visuals.add_scalar('discriminator/loss_D', errors['loss_D'], total_steps + 1)
                # generator
                visuals.add_scalar('generator/loss_G_Attention', errors['loss_G'], total_steps + 1)
                print(errors)
                print('iteration time: %f' % t)

            if epoch_iter % display_freq == 0:
                pred_attention, segmap = model.get_current_visuals()
                image_out = torch.cat([pred_attention, segmap], 0)
                grid_images = torchvision.utils.make_grid(image_out, nrow=pred_attention.shape[0])
                visuals.add_image('Epoch_(%d)_(%d)/image_out' % (epoch, epoch_iter + 1), grid_images, total_steps + 1)

        if epoch % save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks(checkpoints_dir, model_name, hy_config['gpu_ids'], epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, niter + niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

print("Finish!")
