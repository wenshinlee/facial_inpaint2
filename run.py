import time
import tqdm
import torch
import torchvision

import core.utils as utils
from datasets.dataset import get_dataloader
from core.facial_inpaint import FacialInpaint

if __name__ == '__main__':
    hy_config = utils.get_config('./configs/celeba-hq.yaml')
    batch_size = hy_config['batch_size']

    data_loader = get_dataloader(hyperparameters=hy_config)
    model = FacialInpaint(hy_config)

    is_clear_total_steps = True
    if model.is_train:
        if model.continue_train:
            print('Loading pre-trained network for continue train!')
            model.load_networks()
            if model.epoch_count > model.num_train_semantic_net:
                is_clear_total_steps = False
        else:
            print('NOT Loading pre-trained network, training from scratch!')
    else:
        print('Loading pre-trained network for test!')
        model.load_networks()

    if model.is_train:
        visuals = utils.TensorboardVisuals(model.checkpoints_dir, model.experiment_name)
        total_steps = 0
        for epoch in range(model.epoch_count, model.niter + model.niter_decay + 1):
            model.update_learning_rate()

            epoch_start_time = time.time()
            epoch_iter = 0
            for n_iter, data in tqdm.tqdm(enumerate(data_loader, 0), total=len(data_loader),
                                          desc="epoch-->%d" % epoch, ncols=80, leave=False):
                iter_start_time = time.time()
                total_steps += batch_size
                epoch_iter += batch_size

                model.set_input(data)
                model.optimize_parameters(epoch)

                if epoch_iter % model.print_freq == 0:
                    t = round((time.time() - iter_start_time) / batch_size, 4)

                    errors = model.get_current_errors(epoch)
                    for key, value in errors.items():
                        if epoch <= model.num_train_semantic_net:
                            symbol = '<='
                            visuals.add_scalar('dis(epoch {} {})/{}'.format(symbol, model.num_train_semantic_net, key),
                                               value, total_steps + 1)
                        else:
                            symbol = '>'
                            visuals.add_scalar('dis(epoch {} {})/{}'.format(symbol, model.num_train_semantic_net, key),
                                               value, total_steps + 1)

                    print('iteration time: %f' % t)

                if epoch_iter % model.display_freq == 0:
                    if epoch > model.num_train_semantic_net:
                        segmap, pred_segmap, input_image, gt_image, mask, fake_image = model.get_current_visuals(epoch)
                        image = torch.cat([gt_image, input_image, mask, fake_image], 0)
                    else:
                        segmap, pred_segmap, input_image, gt_image, mask = model.get_current_visuals(epoch)
                        image = torch.cat([gt_image, input_image, mask], 0)
                    image = torch.cat([image, segmap, pred_segmap], 0)
                    grid = torchvision.utils.make_grid(image, nrow=input_image.shape[0])

                    if epoch > model.num_train_semantic_net:
                        visuals.add_image('Epoch>15/Epoch_(%d)_(%d)' % (epoch, epoch_iter + 1),
                                          grid, total_steps + 1)
                    else:
                        visuals.add_image('Epoch<=15/Epoch_(%d)_(%d)' % (epoch, epoch_iter + 1),
                                          grid, total_steps + 1)

            if epoch > model.num_train_semantic_net and is_clear_total_steps:
                total_steps = 0
                is_clear_total_steps = False

            if epoch % model.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, model.niter + model.niter_decay, time.time() - epoch_start_time))

        print("Finish train!")

    else:
        for data in tqdm.tqdm(data_loader, total=len(data_loader),
                              desc="test---->>>", ncols=80, leave=False):
            model.set_input(data)
            model.optimize_parameters(model.which_epoch)
        print("Finish test!")
