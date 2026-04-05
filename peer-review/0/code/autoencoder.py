import lightning as L
import torch


class Autoencoder(L.LightningModule):
    """
    Convolutional autoencoder for 9x9 patches with 8 input channels.
    Encoder:
    (B, 8, 9, 9) -> (B, 16, 9, 9) -> (B, 32, 5, 5) -> (B, embedding_size)
    Decoder:
    (B, embedding_size) -> (B, 32, 5, 5) -> (B, 16, 9, 9) -> (B, 8, 9, 9)
    """
    
    def __init__(
        self, optimizer_config=None, n_input_channels=8, patch_size=9, embedding_size=8
    ):
        super().__init__()

        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config
        self.n_input_channels = n_input_channels
        self.patch_size = patch_size
        self.embedding_size = embedding_size

        # Encoder
        # Two convolutional layers followed by ReLU activations
        self.encoder_cnn = torch.nn.Sequential(
            # (B, 8, 9, 9) --> (B, 16, 9, 9)
            torch.nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            # (B, 16, 9, 9) -> (B, 32, 5, 5)
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        # Compute the output size of the encoder to determine the input size for the linear layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_input_channels, patch_size, patch_size)
            encoder_output = self.encoder_cnn(dummy_input)
            self._encoder_output_shape = encoder_output.shape[1:]  # (32, 5, 5)
            encoder_output_size = encoder_output.numel() # (32 * 5 * 5)

        # Fully connected layers for the encoder
        self.encoder_fc = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(encoder_output_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, embedding_size)
        )

        # Decoder
        # Fully connected layers for the decoder
        self.decoder_fc = torch.nn.Sequential(
            # (B, embedding_size) --> (B, 64)
            torch.nn.Linear(embedding_size, 64),
            torch.nn.ReLU(),
            # (B, 64) --> (B, 32 * 5 * 5)
            torch.nn.Linear(64, encoder_output_size),
            torch.nn.ReLU()
        )
        self.unflatten = torch.nn.Unflatten(1, self._encoder_output_shape)

        # ConvTranspose layers
        self.decoder_cnn = torch.nn.Sequential(
            # (B, 32, 5, 5) -> (B, 16, 9, 9)
            torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            # (B, 16, 9, 9) -> (B, 8, 9, 9)
            torch.nn.ConvTranspose2d(16, n_input_channels, kernel_size=3, stride=1, padding=1)
        )

    def encode(self, x):
        h = self.encoder_cnn(x)
        z = self.encoder_fc(h)
        return z

    def decode(self, z):
        h = self.decoder_fc(z)
        h = self.unflatten(h)
        xhat = self.decoder_cnn(h)
        xhat = xhat[:, :, :self.patch_size, :self.patch_size]  # crop to original patch size
        return xhat

    def forward(self, x):
        """
        Run the full autoencoder
        """
        z = self.encode(x)
        return self.decode(z)
    

    def embed(self, x):
        """
        Embeds the input tensor, and return the latent representation
        """
        return self.encode(x)

    def training_step(self, batch):
        """
        Training step for the autoencoder, 
        return the training loss of the autoencoder on the batch
        """
        xhat = self.forward(batch)

        # MSE between the input tensor and the decoded tensor
        loss = torch.nn.functional.mse_loss(xhat, batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        """
        Validation step for the autoencoder, 
        return the validation loss of the autoencoder on the batch
        """
        xhat = self.forward(batch)

        # validation loss (MSE)
        loss = torch.nn.functional.mse_loss(xhat, batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        """ Set up the optimizer """
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        return optimizer

