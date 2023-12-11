import torch
import matplotlib.pyplot as plt
import metric

# draw multiple samples
def sample_figures(num_samples, datasets, labels=None):
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 1 * 3))
    for i, ax in enumerate(axes):
        if labels:
            print(labels[i])
        ax.imshow(datasets[i].detach().squeeze().cpu().numpy(), cmap='gray')
        ax.axis('off')

    plt.show()


# draw one sample
def sample_figure(dataset):
    fig, axes = plt.subplots(1, 1, figsize=(1 * 3, 1 * 3))
    ax = axes
    ax.imshow(dataset.detach().squeeze().cpu().numpy(), cmap='gray')
    ax.axis('off')
    
    plt.show()

def result_figure(user_choice, gen_img, denoised_img):
    # user select image, generated image, generated image denoised with DAE
    plt.subplot(1,3,1)
    plt.imshow(user_choice.detach().squeeze().cpu().numpy(), cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(gen_img.detach().squeeze().cpu().numpy(), cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(torch.squeeze(denoised_img).detach().cpu().numpy().reshape(28, 28),cmap='gray')

    plt.show()

def maximum_similarity_result_figure(dae_model, similarity_list, x_list, chosen_r_dataset, device):
    # Visualize the DDPM iteration output with Maximum similarity
    # user select image, generated image, generated image denoised with DAE
    max_cos_index = similarity_list.index(max(similarity_list))

    print(f'maximum similarity DDPM iteration: {max_cos_index}')

    gen_img = x_list[max_cos_index].to('cpu').apply_(lambda x: (x + 1) / 2).unsqueeze(dim = 1).to(device) #  [-1, 1] -> [0, 1]

    denoised_img_z, denoised_img = dae_model(gen_img) # DAE without noise

    result_figure(chosen_r_dataset, gen_img, denoised_img)

    return denoised_img

