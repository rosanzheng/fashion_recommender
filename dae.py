import torch.nn as nn

class DAE_CNN(nn.Module):
    def __init__(self, c, latent_dim):
        super(DAE_CNN,self).__init__()

        self.c = c
        encoder_layers = []
        decoder_layers = []

        encoder_layers.append(nn.Conv2d(in_channels=1, out_channels=self.c, kernel_size=4, stride=2, padding=1))
        encoder_layers.append(nn.ReLU())
    
        encoder_layers.append(nn.Conv2d(in_channels=self.c, out_channels=self.c * 2, kernel_size=4, stride=2, padding=1))
        encoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)

        self.encoder_fc = nn.Linear(self.c * 2 * 7 * 7, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, self.c * 2 * 7 * 7)
        
        decoder_layers.append(nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.ReLU())

        decoder_layers.append(nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)
                

                
    def forward(self, x):                # x: (batch_size, 1, 28, 28)
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        encoder_output = self.encoder_fc(z) # encoder output: (batch_size, latent_dim)
        
        z = self.decoder_fc(encoder_output)
        z = z.view(z.size(0), self.c * 2, 7, 7)
        out = self.decoder(z)
        return encoder_output, out
