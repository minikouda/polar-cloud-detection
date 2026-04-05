import lightning as L
import torch


class Autoencoder(L.LightningModule):
    def __init__(
        self,
        optimizer_config=None,
        n_input_channels=8,
        patch_size=9,
        embedding_size=8,
        model_type="mlp",
        hidden_dims=None,
        conv_channels=None,
        activation="relu",
        dropout_rate=0.0,
        denoising=False,
        noise_std=0.0,
        images_padded=None,
    ):
        """
        Initializes the autoencoder model.
        Args:
            optimizer_config: A dictionary containing optimizer configuration parameters.
            embedding_size: The size of the latent embedding space.
            model_type: The type of autoencoder to build, either "mlp" or "conv".
            hidden_dims: A list of hidden dimensions for the MLP autoencoder (ignored for conv autoencoder).
            conv_channels: A list of output channels for each convolutional layer in the convolutional autoencoder (ignored for mlp autoencoder).
            activation: The activation function to use in the autoencoder layers, either "relu", "gelu", or "leaky_relu".
            dropout_rate: The dropout rate to use in the autoencoder layers.
            denoising: Whether to use a denoising autoencoder
            noise_std: The standard deviation of the Gaussian noise.
            images_padded: A tensor of shape (n_images, n_input_channels, padded_width, padded_height) containing the padded input images. Required if using coordinate-based input batches.
        """
        super().__init__()

        # set up optimizer config and model parameters based on input arguments
        if optimizer_config is None:
            optimizer_config = {}
        self.optimizer_config = optimizer_config
        self.pad_len = patch_size // 2
        self.denoising = denoising
        self.noise_std = noise_std
        self.model_type = model_type
        self.n_input_channels = n_input_channels
        self.patch_size = patch_size
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate

        # register buffers for coordinate indexing and image storage
        self.register_buffer(
            "dy", torch.arange(-self.pad_len, self.pad_len + 1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "dx", torch.arange(-self.pad_len, self.pad_len + 1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "channel_idx",
            torch.arange(n_input_channels, dtype=torch.long)[None, :, None, None],
            persistent=False,
        )
        if images_padded is not None:
            self.register_buffer(
                "images_padded",
                torch.as_tensor(images_padded, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.images_padded = None

        # build autoencoder structure based on config
        if hidden_dims is None:
            hidden_dims = [128, 64]
        if conv_channels is None:
            conv_channels = [32, 64]
        if model_type == "conv":
            self._build_conv_autoencoder(
                n_input_channels=n_input_channels,
                patch_size=patch_size,
                embedding_size=embedding_size,
                conv_channels=conv_channels,
                activation=activation,
                dropout_rate=dropout_rate,
            )
        else:
            self._build_mlp_autoencoder(
                n_input_channels=n_input_channels,
                patch_size=patch_size,
                embedding_size=embedding_size,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout_rate=dropout_rate,
            )

    def _get_activation(self, activation):
        """
        return the activation function module based on the input string.
        """
        if activation == "gelu":
            return torch.nn.GELU()
        if activation == "leaky_relu":
            return torch.nn.LeakyReLU()
        return torch.nn.ReLU()

    def _build_mlp_autoencoder(
        self,
        n_input_channels,
        patch_size,
        embedding_size,
        hidden_dims,
        activation,
        dropout_rate,
    ):
        """
        Build the MLP autoencoder.
        """
        input_size = int(n_input_channels * (patch_size**2))

        # build encoder layers for mlp autoencoder
        encoder_layers = [torch.nn.Flatten(start_dim=1, end_dim=-1)]
        prev_dim = input_size
        for hidden_dim in hidden_dims:
            encoder_layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(self._get_activation(activation))
            if dropout_rate > 0:
                encoder_layers.append(torch.nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        encoder_layers.append(torch.nn.Linear(prev_dim, embedding_size))
        self.encoder = torch.nn.Sequential(*encoder_layers)

        # build decoder layers for mlp autoencoders
        decoder_layers = []
        rev_hidden = list(hidden_dims)[::-1]
        prev_dim = embedding_size
        for hidden_dim in rev_hidden:
            decoder_layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(self._get_activation(activation))
            if dropout_rate > 0:
                decoder_layers.append(torch.nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        decoder_layers.append(torch.nn.Linear(prev_dim, input_size))
        decoder_layers.append(torch.nn.Unflatten(1, (n_input_channels, patch_size, patch_size)))
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def _build_conv_autoencoder(
        self,
        n_input_channels,
        patch_size,
        embedding_size,
        conv_channels,
        activation,
        dropout_rate,
    ):
        """
        Build the convolutional autoencoder.
        """
        # build encoder layers for convolutional autoencoder
        enc_layers = []
        in_ch = n_input_channels
        for out_ch in conv_channels:
            enc_layers.append(torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            enc_layers.append(self._get_activation(activation))
            if dropout_rate > 0:
                enc_layers.append(torch.nn.Dropout2d(dropout_rate))
            in_ch = out_ch
        self.conv_encoder = torch.nn.Sequential(*enc_layers)
        self.conv_out_channels = in_ch
        self.conv_flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.to_embedding = torch.nn.Linear(self.conv_out_channels * patch_size * patch_size, embedding_size)

        # build decoder layers for convolutional autoencoder
        self.from_embedding = torch.nn.Linear(embedding_size, self.conv_out_channels * patch_size * patch_size)
        dec_layers = []
        rev_channels = list(conv_channels)[::-1]
        for i in range(len(rev_channels)):
            curr_ch = rev_channels[i]
            if i + 1 < len(rev_channels):
                next_ch = rev_channels[i + 1]
            else:
                next_ch = n_input_channels
            dec_layers.append(
                torch.nn.ConvTranspose2d(curr_ch, next_ch, kernel_size=3, padding=1)
            )
            if i < len(rev_channels) - 1:
                dec_layers.append(self._get_activation(activation))
                if dropout_rate > 0:
                    dec_layers.append(torch.nn.Dropout2d(dropout_rate))
        self.conv_decoder = torch.nn.Sequential(*dec_layers)

    def forward(self, batch):
        """
        Forward pass through the network.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, n_input_channels, width, height)
        """
        batch = self._prepare_batch(batch)
        encoded = self._encode(batch)
        decoded = self._decode(encoded)
        return decoded

    def _encode(self, batch):
        if self.model_type == "conv":
            x = self.conv_encoder(batch)
            x = self.conv_flatten(x)
            x = self.to_embedding(x)
            return x
        return self.encoder(batch)

    def _decode(self, embedding):
        if self.model_type == "conv":
            x = self.from_embedding(embedding)
            x = x.view(
                -1,
                self.conv_out_channels,
                self.patch_size,
                self.patch_size,
            )
            x = self.conv_decoder(x)
            return x
        return self.decoder(embedding)

    def _prepare_batch(self, batch):
        if torch.is_tensor(batch):
            return batch

        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            if self.images_padded is None:
                raise ValueError("images_padded is required when batch contains coordinates")

            image_idx, y_rel, x_rel = batch
            image_idx = image_idx.long()
            y_rel = y_rel.long()
            x_rel = x_rel.long()

            y = y_rel + self.pad_len
            x = x_rel + self.pad_len
            yy = y[:, None, None] + self.dy[None, :, None]
            xx = x[:, None, None] + self.dx[None, None, :]

            patches = self.images_padded[
                image_idx[:, None, None, None],
                self.channel_idx,
                yy[:, None, :, :],
                xx[:, None, :, :],
            ]
            return patches

        return batch

    def training_step(self, batch, batch_idx):
        """
        Training step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The training loss of the autoencoder on the batch
        """
        batch = self._prepare_batch(batch)
        model_input = batch
        if self.denoising and self.noise_std > 0:
            noise = torch.randn_like(batch) * self.noise_std
            model_input = batch + noise

        encoded = self._encode(model_input)
        decoded = self._decode(encoded)
        loss = torch.nn.functional.mse_loss(batch, decoded)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the autoencoder.
        Args:
            batch: A tensor of shape (batch_size, n_input_channels, width, height)
            batch_idx: The index of the batch
        Returns:
            The validation loss of the autoencoder on the batch
        """
        batch = self._prepare_batch(batch)
        encoded = self._encode(batch)
        decoded = self._decode(encoded)
        loss = torch.nn.functional.mse_loss(batch, decoded)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        # set up the optimizer.
        optimizer_cfg = dict(self.optimizer_config)
        optimizer_name = optimizer_cfg.pop("name", "adam").lower()

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), **optimizer_cfg)
        else:
            optimizer = torch.optim.Adam(self.parameters(), **optimizer_cfg)
        return optimizer

    def embed(self, x):
        """
        Embeds the input tensor.
        Args:
            x: A tensor of shape (batch_size, n_input_channels, width, height)
        Returns:
            A tensor of shape (batch_size, embedding_size)
        """
        return self._encode(x)
