import torch
import torch.nn.functional as F
from tqdm import tqdm

import utils

# Calculate euclidean_distance
def euclidean_distance(a, b):
    # Calculate the difference between two points.
    diff = a - b
    
    # Calculate the square of each dimension.
    squared_diff = diff ** 2
    
    # Find the sum for each dimension.
    sum_squared_diff = torch.sum(squared_diff)
    
    # Take the root and calculate the Euclidean distance.
    distance = torch.sqrt(sum_squared_diff)
    
    return distance

def calculate_similarity(fake_image, real_image):
    
    fake_image = fake_image.flatten().detach().cpu()
    real_image = real_image.flatten().detach().cpu()

    cosine_sim = F.cosine_similarity(fake_image, real_image, dim=0)

    # euc distance
    euc = euclidean_distance(fake_image, real_image)
    
    return cosine_sim, euc

def calculate_DAE_similarity(dae_model, chosen_r_dataset, x_list, device):
    similarity_list = []
    euc_list = []

    for x_img in x_list:
        gen_img = x_img.to('cpu').apply_(lambda x: (x + 1) / 2).unsqueeze(dim = 1) # [-1, 1] -> [0, 1]

        real_x = chosen_r_dataset.unsqueeze(dim=0).to(device)
        real_z, real_out = dae_model(real_x)
    #
        fake_x = gen_img.to(device)
        fake_z, fake_out = dae_model(fake_x)
        
        fake_z = fake_z.flatten()
        real_z = real_z.flatten()

        # cosine_sim
        cosine_sim = F.cosine_similarity(fake_z, real_z, dim=0)
        similarity_list.append(cosine_sim.item())

        # euc distance
        euc = euclidean_distance(fake_z, real_z)
        euc_list.append(euc.item())

    return similarity_list, euc_list
    
def calculate_real_similarity(dataloaders, dae_model, denoised_img, noise_ratio, device, label_num):
    # calculate similarity between real image and generated image with DDPM

    real_similarity_list = []
    real_euc_list = []
    real_image_list = []


    with torch.no_grad():


        running_loss = 0.0
        for inputs, label in tqdm(dataloaders["train"]):
            # Calculate similarity only for items with the same label
            if label == label_num:
                # DDPM output denoised with DAE + noise
                noisy_fake_x = utils.add_noise(denoised_img, noise_ratio, device).to(device)
                fake_z, fake_out = dae_model(noisy_fake_x)
                
                # real image + noise
                noisy_x = utils.add_noise(inputs, noise_ratio, device).to(device) 
                inputs = inputs.to(device)                                       

                z, out = dae_model(noisy_x)
                real_image_list.append(inputs.cpu())
                
                # cosine_sim
                cosine_sim = F.cosine_similarity(fake_z, z, dim=1)
                real_similarity_list.append(cosine_sim.item())

                # euc distance
                euc = euclidean_distance(fake_z, z)
                real_euc_list.append(euc.item())
                

    real_image_list = torch.cat(real_image_list, dim = 0)

    return real_image_list, real_similarity_list, real_euc_list