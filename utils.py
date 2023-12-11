import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import random
from tqdm import tqdm


import time
import copy

import metric
import heapq


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

# define function
def transform_fuc(examples):
   # define image transformations (e.g. using torchvision)
    transform = transforms.Compose([
                transforms.ToTensor(),
    ])

    examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
    del examples["image"]

    return examples

# diffusion forward process q
def get_noisy_image(diffusion, x_start, t): 
  
  # add noise
  x_noisy = diffusion.q_sample(x_start, t=t)

  return x_noisy

# regenerate image with DDPM
def regenerate_image(diffusion, x_start, device, t_num = 200, batch_size = 10):
    x_start = x_start.to(device) 

    t = torch.tensor([t_num] * batch_size).to(device) # diffusion timestep

    x = get_noisy_image(diffusion, x_start, t) # diffusion forward process q

    x_p = x.unsqueeze(dim = 1)
    x_start_p = x_start.unsqueeze(dim=1)
    
    # diffusion reverse process p
    for i in tqdm(reversed(range(t_num)), total = t_num):
        x_p, x_start_p = diffusion.p_sample(x_p, i)

    return x_start, x_p, x_start_p #x_{p-1}, x_p, x_p


# add noise for DAE
def add_noise(img, ratio, device):
    noise = torch.randn(img.size()).to(device) * ratio 
    noisy_img = img.to(device) + noise
    return noisy_img

# Select the image with the highest similarity to the user-selected item among ddpm regenerate images.
def compare_chosen_generated(start_index, end_index, chosen_r_dataset, x_p, device): 


    max_cosine_sim = 0
    max_index = start_index

    for idx in range(start_index, end_index):
        real_gen_dataset = torch.cat((chosen_r_dataset.to(device), x_p[idx].to(device)), dim = 0)
        real_gen_dataset.size()

        num_samples = real_gen_dataset.size(0)

        fake_image = x_p[idx].flatten().detach().cpu()
        real_image = chosen_r_dataset.flatten().detach().cpu()

        cosine_sim, euc = metric.calculate_similarity(fake_image, real_image)

        if cosine_sim >= max_cosine_sim:
            max_cosine_sim = cosine_sim
            max_index = idx
            max_real_gen_dataset = real_gen_dataset

    return x_p[idx]

# Function to find index with top_n value
def top_n_indexes(arr, n):
    min_heap = [(value, index) for index, value in enumerate(arr[:n])]
    heapq.heapify(min_heap)

    for i, value in enumerate(arr[n:], start=n):
        if value > min_heap[0][0]:
            heapq.heappop(min_heap)
            heapq.heappush(min_heap, (value, i))

    top_n_indexes = [index for (_, index) in sorted(min_heap, reverse=True)]
    return top_n_indexes