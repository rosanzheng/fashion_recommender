import torch

import time
import copy

import utils


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10, noise_ratio = 0.1):
    since = time.time()

    train_loss_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_loss = 100000000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()            # Set model to training mode
            else:
                model.eval()            # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                noisy_x = utils.add_noise(inputs, noise_ratio, device).to(device)
                                          
                y = inputs.view(-1, 1, 28, 28).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    z, out = model(noisy_x)
                    loss = criterion(out, y)  # calculate a loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()                             # perform back-propagation from the loss
                        optimizer.step()                             # perform gradient descent with given optimizer

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
            # deep copy the model
            if phase == 'train':
                train_loss_history.append(epoch_loss)


            if phase == 'train' and epoch_loss < best_train_loss:
                best_train_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history