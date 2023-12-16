# DS535_Fashion_Recommender

## Dataset
- FashionMNIST

### Model Weight 
- DDPM: DDPM_checkpoint.pth
- DAE: DAE_CNN_checkpoint.pth

[Model Weight Download Link](https://drive.google.com/drive/folders/1IeV8rfYLovpuPNf4tHZUvBjGe5P4e2px?usp=sharing)

After downloading Model Weight from google drive, save Model Weight File in ckpts folder.

-----
## Model
- DDPM: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- DAE: [Extracting and Composing Robust Features with Denoising
Autoencoders](https://dl.acm.org/doi/10.1145/1390156.1390294)
-----
## Project Structure

```

├── ckpts
│
├── FashionMNIST_DDPM.ipynb 
├── trainer.py
├── visualize.py
├── metric.py
├── dae.py
└── utils.py
```

| Folder       | Usage                          |
| ------------ | ------------------------------ |
| ckpts        | save checkpoint model          |
-----
## Environment
```
    # conda pip upgrade
    conda update --all
    python.exe -m pip install --upgrade pip
    
    # conda virtual environment
    conda env update --file requirements.yml
    conda activate DS535_Fashion

    # add jupyter kernel
    python -m ipykernel install --user --name=DS535_Fashion

    # Restart VSCode or Jupyter Notebook
```
-----
## User Guide
### FashionMNIST_DDPM.ipynb
1. Download DDPM_checkpoint.pth or Train DDPM

2. Choose one User Choice Item From Real FashionMNIST

<p align="center"><img src=image-7.png height="100px" width="100px"></p>

3. Generate Fake Image From Noise Z with DDPM



4. Choose Generated Image similar with chosen image from Real data, and regenerate
    - If an image similar to the real item selected in the  step 2 is not generated, go back to step 3 and regenerate the Fake Image.


<p align="center"><img src=image-8.png height="100px" width="100px"></p>


5. Regenerate until regenerated image from noise similar to chosen image (DDPM iteration)

<p align="center"><img src=image-3.png height="100px" width="100px"></p>
6. Denoise Regenerated Image with DAE(with CNN layer)
    - Download DAE_CNN_checkpoint.pth or Train DAE

<p align="center"><img src=image-9.png height="100px" width="100px"></p>

7. Find the real top 10 items closest to the generated item

<p align="center"><img src=image-6.png height="80px" width="1000px"></p>


