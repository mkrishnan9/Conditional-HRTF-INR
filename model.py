
# This code is based on the implementation of the DiGS model and loss functions
# It was partly based on SIREN and SAL implementation and architecture but with several significant modifications.
# for the original SIREN version see: https://github.com/vsitzmann/siren
# for the original SAL version see: https://github.com/matanatz/SAL

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import utils as utils
from torchmeta.modules.module import MetaModule
from torchmeta.modules.container import MetaSequential
from torchmeta.modules.utils import get_subdict
from collections import OrderedDict

################################# New Conditioning Methods #################################




class PositionalEncoder(nn.Module):
    """
    A module to apply sinusoidal positional encoding to input coordinates.
    """
    def __init__(self, d_input: int, n_freqs: int):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.d_output = d_input * (2 * n_freqs)

        # Create a buffer for frequency bands that is not a model parameter
        freq_bands = 2.**torch.linspace(0., n_freqs - 1, n_freqs)
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies positional encoding to the input tensor.
        Args:
            x: Input tensor of shape (..., d_input)
        Returns:
            Encoded tensor of shape (..., d_output)
        """
        # Project input coordinates onto frequency bands
        # Shape: (..., d_input, n_freqs)
        x_proj = x.unsqueeze(-1) * self.freq_bands

        # Concatenate sine and cosine transformations
        # Shape: (..., d_input * 2 * n_freqs)
        x_encoded = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        return x_encoded.reshape(*x.shape[:-1], self.d_output)


class CondHRTFNetwork(nn.Module):
    """
    An INR for HRTFs, conditioned on anthropometry using an MLP.

    This network follows the Attn> Concatenation paper's best-practice for MLP conditioning:
    1. It uses an embedding network to create a latent vector `z` from anthropometry.
    2. It splits `z` into N chunks.
    3. It concatenates each chunk to the input of one of the N hidden layers of the main INR.
    Note: Embedding layer dimensions are fixed at 256. Need to add them as an argument.
    """
    def __init__(self,
                 d_anthro_in: int = 19,
                 d_latent: int = 256,
                 d_inr_hidden: int = 256,
                 n_inr_layers: int = 4,
                 d_inr_out: int = 92,
                 n_freqs: int = 10,
                 d_embed_hidden = 256,
                 d_inr_in = 2):
        super().__init__()

        self.d_inr_in = d_inr_in # Azimuth and Elevation
        self.d_latent = d_latent
        self.n_inr_layers = n_inr_layers

        # Ensure the latent vector can be split evenly among the layers
        assert d_latent % n_inr_layers == 0, \
            "Latent dimension (d_latent) must be divisible by number of layers (n_inr_layers)."
        self.d_latent_split = d_latent // n_inr_layers

        # 1. Positional Encoder for coordinates (azi, ele)
        self.positional_encoder = PositionalEncoder(self.d_inr_in, n_freqs)
        d_pe_out = self.positional_encoder.d_output

        # 2. Embedding network to convert anthropometry to a latent vector `z`
        self.embedding_net = nn.Sequential(
            nn.Linear(d_anthro_in, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, d_latent)
        )

        # 3. Main INR network with layer-wise conditioning
        self.main_inr_net = nn.ModuleList()

        # First layer
        self.main_inr_net.append(
            nn.Linear(d_pe_out + self.d_latent_split, d_inr_hidden)
        )

        # Hidden layers
        for _ in range(n_inr_layers - 1):
            self.main_inr_net.append(
                nn.Linear(d_inr_hidden + self.d_latent_split, d_inr_hidden)
            )

        # Output layer
        self.output_layer = nn.Linear(d_inr_hidden, d_inr_out)

    def forward(self, coords: torch.Tensor, anthropometry: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass.
        Args:
            coords: A tensor of (azimuth, elevation) coordinates, shape (B, 2).
            anthropometry: A tensor of anthropometric data, shape (B, d_anthro_in).
        Returns:
            A tensor of HRTF values (e.g., mag, phase), shape (B, d_inr_out).
        """
        # 1. Generate the latent vector `z` from anthropometry
        z = self.embedding_net(anthropometry)

        # 2. Split `z` into one chunk per layer
        z_splits = torch.chunk(z, self.n_inr_layers, dim=-1)

        # 3. Apply positional encoding to coordinates
        x = self.positional_encoder(coords)

        # 4. Pass through the main network, conditioning each layer
        for i, layer in enumerate(self.main_inr_net):
            # Concatenate the signal with the i-th chunk of the latent vector
            x = torch.cat([x, z_splits[i]], dim=-1)
            x = torch.relu(layer(x))

        # 5. Final output layer
        output = self.output_layer(x)
        return output


##To prevent overfitting, tried adding dropout to the CondHRTFNetwork class. Not sure if I did it right.

class CondHRTFNetwork_with_Dropout(nn.Module):
    """
    An INR for HRTFs with dropout for regularization.

    This version adds dropout layers to:
    1. The embedding network that creates the latent vector `z`.
    2. The main INR network's hidden layers.
    """
    def __init__(self,
                 d_anthro_in: int = 19,
                 d_latent: int = 256,
                 d_inr_hidden: int = 256,
                 n_inr_layers: int = 4,
                 d_inr_out: int = 92,
                 n_freqs: int = 10,
                 dropout_rate: float = 0.5): # <-- New parameter for dropout
        super().__init__()

        self.d_inr_in = 2 # Azimuth and Elevation
        self.d_latent = d_latent
        self.n_inr_layers = n_inr_layers

        # Ensure latent vector can be split evenly
        assert d_latent % n_inr_layers == 0, \
            "Latent dimension (d_latent) must be divisible by number of layers (n_inr_layers)."
        self.d_latent_split = d_latent // n_inr_layers

        # 1. Positional Encoder
        self.positional_encoder = PositionalEncoder(self.d_inr_in, n_freqs)
        d_pe_out = self.positional_encoder.d_output

        # 2. Embedding network with dropout
        self.embedding_net = nn.Sequential(
            nn.Linear(d_anthro_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, d_latent)
        )

        # 3. Main INR network
        self.main_inr_net = nn.ModuleList()

        # First layer
        self.main_inr_net.append(
            nn.Linear(d_pe_out + self.d_latent_split, d_inr_hidden)
        )

        # Hidden layers
        for _ in range(n_inr_layers - 1):
            self.main_inr_net.append(
                nn.Linear(d_inr_hidden + self.d_latent_split, d_inr_hidden)
            )

        # Output layer
        self.output_layer = nn.Linear(d_inr_hidden, d_inr_out)

        # 4. Dropout layer for the main INR network
        self.inr_dropout = nn.Dropout(dropout_rate)

    def forward(self, coords: torch.Tensor, anthropometry: torch.Tensor) -> torch.Tensor:
        z = self.embedding_net(anthropometry)
        z_splits = torch.chunk(z, self.n_inr_layers, dim=-1)
        x = self.positional_encoder(coords)
        for i, layer in enumerate(self.main_inr_net):
            x = torch.cat([x, z_splits[i]], dim=-1)
            x = torch.relu(layer(x))
            x = self.inr_dropout(x) # <-- Dropout applied


        output = self.output_layer(x)
        return output

## Tried adding convolutional layer to understand relationship across frequencies for the R^92 output vector.
class Conv1DRefiner(nn.Module):
    def __init__(self, embed_dim, num_layers=3, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim,
                                    kernel_size=kernel_size, padding=padding))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm([embed_dim, 92])) ##92 Frequency bins

        self.refiner = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim) -> (N, 92, 32)
        x = x.permute(0, 2, 1) # -> (N, 32, 92)

        refined_x = self.refiner(x)

        # Permute back to the original dimension order for the final head
        refined_x = refined_x.permute(0, 2, 1) # -> (N, 92, 32)
        return refined_x



class CondHRTFNetwork_conv(nn.Module):
    """
    An INR for HRTFs that uses a "refinement" architecture.
    An MLP backbone predicts a feature vector for each frequency, which is then
    refined by a 1D CNN before final projection.
    """
    def __init__(self,
                 d_anthro_in: int = 19,
                 d_latent: int = 256,
                 d_inr_hidden: int = 256,
                 n_inr_layers: int = 4,
                 d_inr_out: int = 92, # This is now the sequence length
                 n_freqs: int = 10,
                 d_freq_embed: int = 16): # New: Dimension for each frequency's feature vector
        super().__init__()

        self.d_inr_in = 2
        self.d_latent = d_latent
        self.n_inr_layers = n_inr_layers
        self.d_inr_out = d_inr_out
        self.d_freq_embed = d_freq_embed

        assert d_latent % n_inr_layers == 0, \
            "Latent dimension must be divisible by number of layers."
        self.d_latent_split = d_latent // n_inr_layers

        self.positional_encoder = PositionalEncoder(self.d_inr_in, n_freqs)
        d_pe_out = self.positional_encoder.d_output

        self.embedding_net = nn.Sequential(
            nn.Linear(d_anthro_in, 256), nn.ReLU(),
            nn.Linear(256,256), nn.ReLU(),
            nn.Linear(256, d_latent)
        )


        self.main_inr_net = nn.ModuleList()
        self.main_inr_net.append(nn.Linear(d_pe_out + self.d_latent_split, d_inr_hidden))
        for _ in range(n_inr_layers - 1):
            self.main_inr_net.append(nn.Linear(d_inr_hidden + self.d_latent_split, d_inr_hidden))



        # This predicts a feature vector for every frequency bin.
        self.initial_predictor = nn.Linear(d_inr_hidden, d_inr_out * d_freq_embed)

        # Convolutional Refiner
        self.refiner = Conv1DRefiner(embed_dim=d_freq_embed, num_layers=3, kernel_size=7)

        # This small MLP maps each refined feature vector to a single output value.
        self.final_head = nn.Sequential(
            nn.Linear(d_freq_embed, d_freq_embed // 2),
            nn.GELU(),
            nn.Linear(d_freq_embed // 2, 1)
        )

    def forward(self, coords: torch.Tensor, anthropometry: torch.Tensor) -> torch.Tensor:
        batch_size = coords.shape[0]


        z = self.embedding_net(anthropometry)
        z_splits = torch.chunk(z, self.n_inr_layers, dim=-1)
        x_pe = self.positional_encoder(coords)
        x = x_pe
        for i, layer in enumerate(self.main_inr_net):
            x = torch.cat([x, z_splits[i]], dim=-1)
            x = torch.relu(layer(x))

        x = self.initial_predictor(x)
        x = x.view(batch_size, self.d_inr_out, self.d_freq_embed)


        x = self.refiner(x)


        output = self.final_head(x)


        return output.squeeze(-1)

########FiLM for conditioning###########

class FiLMLayer(nn.Module):
    """
    A single layer that applies Feature-wise Linear Modulation.
    It is modulated by gamma/beta parameters.
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.layer = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Applies the FiLM operation.
        Args:
            x: The main signal tensor.
            gamma: The scale parameter tensor.
            beta: The shift parameter tensor.
        """
        x = self.layer(x)
        return gamma * x + beta


class CondHRTFNetwork_FiLM(nn.Module):

    def __init__(self,
                 d_anthro_in: int = 15,
                 d_latent: int = 512,
                 d_inr_hidden: int = 256,
                 n_inr_layers: int = 8,
                 d_inr_out: int = 92,
                 n_freqs: int = 10):
        super().__init__()

        self.d_inr_in = 2
        self.n_inr_layers = n_inr_layers
        assert d_latent % n_inr_layers == 0, \
            "Latent dimension (d_latent) must be divisible by number of layers (n_inr_layers)."
        self.d_latent_split = d_latent // n_inr_layers

        self.positional_encoder = PositionalEncoder(self.d_inr_in, n_freqs)
        d_pe_out = self.positional_encoder.d_output


        self.embedding_net = nn.Sequential(
            nn.Linear(d_anthro_in, 128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128, d_latent)
        )


        self.main_inr_net = nn.ModuleList()
        self.main_inr_net.append(FiLMLayer(d_pe_out, d_inr_hidden))
        for _ in range(n_inr_layers - 1):
            self.main_inr_net.append(FiLMLayer(d_inr_hidden, d_inr_hidden))


        self.film_generator = nn.ModuleList()
        for _ in range(n_inr_layers):
            self.film_generator.append(
                nn.Sequential(
                    nn.Linear(self.d_latent_split, 128),
                    nn.ReLU(),
                    nn.Linear(128, d_inr_hidden * 2)
                )
            )


        self.output_layer = nn.Linear(d_inr_hidden, d_inr_out)

    def forward(self, coords: torch.Tensor, anthropometry: torch.Tensor) -> torch.Tensor:

        z = self.embedding_net(anthropometry)
        z_splits = torch.chunk(z, self.n_inr_layers, dim=-1)
        x = self.positional_encoder(coords)

        for i, film_layer in enumerate(self.main_inr_net):
            gamma_beta = self.film_generator[i](z_splits[i])
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            x = film_layer(x, gamma, beta)
            x = torch.relu(x)


        output = self.output_layer(x)
        return output


#########Cross-attention Conditioning#####

class CrossAttentionStage(nn.Module):
    """
    A single stage of the attention-based decoder, as described in the paper.
    It consists of a cross-attention layer followed by a multi-layer perceptron (MLP),
    [cite_start]with skip connections and layer normalization. [cite: 555]
    """
    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()


        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        #3-layer MLP that follows the attention layer
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model)
        )

        # Layer normalization, applied after skip connections
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The main signal tensor (query), from coordinates. Shape: [batch, n_coords, d_model]
            z (torch.Tensor): The set-latent conditioning tensor (key, value). Shape: [batch, n_tokens, d_token]
        """
        #  Cross-Attention
        # The query is derived from the coordinates.
        # key and value are derived from the set-latent tokens (z).
        attn_output, _ = self.cross_attention(query=x, key=z, value=z)

        #Skip connection and normalization
        x = self.norm1(x + attn_output)

        # MLP
        mlp_output = self.mlp(x)

        #Second skip connection and normalization
        x = self.norm2(x + mlp_output)

        return x


class CondHRTFNetwork_Attention(nn.Module):
    """
    Full attention-based neural field, conditioned on a set-latent representation.
    """
    def __init__(self,
                 d_anthro_in: int = 15,
                 d_model: int = 128,
                 d_mlp: int = 256,
                 d_inr_out: int = 92,
                 n_freqs: int = 10,
                 n_stages: int = 3,
                 n_heads: int = 8,
                 n_tokens: int = 16):
        super().__init__()

        self.d_inr_in = 2 # 2D coordinates (e.g., azimuth, elevation)
        self.n_tokens = n_tokens
        self.d_model = d_model

        # 1. Positional Encoder for coordinates
        self.positional_encoder = PositionalEncoder(self.d_inr_in, n_freqs)
        d_pe_out = self.positional_encoder.d_output

        # Layer to project positional encoding to the model's dimension
        self.coord_projection = nn.Linear(d_pe_out, d_model)

        # 2. Embedding network to generate the set-latent vector `z`
        self.embedding_net = nn.Sequential(
            nn.Linear(d_anthro_in, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_tokens * self.d_model) # Output enough values for all tokens
        )

        # main network, composed of multiple attention stages
        self.attention_stages = nn.ModuleList(
            [CrossAttentionStage(d_model, n_heads, d_mlp) for _ in range(n_stages)]
        )

        #Final output layer
        self.output_layer = nn.Linear(d_model, d_inr_out)

    def forward(self, coords: torch.Tensor, anthropometry: torch.Tensor) -> torch.Tensor:
        z_flat = self.embedding_net(anthropometry)
        z = z_flat.view(-1, self.n_tokens, self.d_model)
        x_pe = self.positional_encoder(coords)
        batch_size = z.shape[0]
        total_coords = coords.shape[0]
        n_locations = total_coords // batch_size
        x = self.coord_projection(x_pe)


        # This ensures x has the shape (batch_size, n_locations, d_model)
        if x.dim() == 2:
            x = x.view(batch_size, n_locations, self.d_model)

        for stage in self.attention_stages:
            x = stage(x, z)

        output = self.output_layer(x)
        return output


###### Things taken from DiGS paper#############

class Decoder(nn.Module):

    def forward(self, *args, **kwargs):
        return self.fc_block(*args, **kwargs)

class DiGSNetwork(nn.Module):
    def __init__(self, latent_size, in_dim=3, decoder_hidden_dim=256, nl='sine', encoder_type=None,
                 decoder_n_hidden_layers=8, init_type='siren', sphere_init_params=[1.6, 1.0]):
        super().__init__()
        self.encoder_type = encoder_type
        self.init_type = init_type
        if encoder_type == 'autodecoder':
            # latent_size will stay as input latent size
            pass
        elif encoder_type == 'none':
            latent_size = 0
        else:
            raise ValueError("unsupported encoder type")
        self.decoder = Decoder()
        self.decoder.fc_block = FCBlock(in_dim + latent_size, 1, num_hidden_layers=decoder_n_hidden_layers, hidden_features=decoder_hidden_dim,
                                outermost_linear=True, nonlinearity=nl, init_type=init_type,
                                sphere_init_params=sphere_init_params)  # SIREN decoder

    def forward(self, non_mnfld_pnts, mnfld_pnts=None):
        # shape is (bs, npoints, in_dim+latent_size) for both inputs, npoints could be different sizes
        batch_size = non_mnfld_pnts.shape[0]
        if not mnfld_pnts is None and self.encoder_type == 'autodecoder':
            # Assume inputs have latent vector concatted with [xyz, latent]
            latent = non_mnfld_pnts[:,:,3:]
            latent_reg = latent.norm(dim=-1).mean()
            manifold_pnts_pred = self.decoder(mnfld_pnts.view(-1, mnfld_pnts.shape[-1])).reshape(batch_size, -1)
        elif mnfld_pnts is not None:
            manifold_pnts_pred = self.decoder(mnfld_pnts.view(-1, mnfld_pnts.shape[-1])).reshape(batch_size, -1)
            latent = None
            latent_reg = None
        else:
            manifold_pnts_pred = None
            latent = None
            latent_reg = None

        # Off manifold points
        nonmanifold_pnts_pred = self.decoder(non_mnfld_pnts.view(-1, non_mnfld_pnts.shape[-1])).reshape(batch_size, -1)

        return {"manifold_pnts_pred": manifold_pnts_pred,
                "nonmanifold_pnts_pred": nonmanifold_pnts_pred,
                "latent_reg": latent_reg,
                "latent": latent}


class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', init_type='siren',
                 sphere_init_params=[1.6,1.0]):
        super().__init__()
        print("decoder initialising with {} and {}".format(nonlinearity, init_type))

        self.first_layer_init = None
        self.sphere_init_params = sphere_init_params
        self.init_type = init_type

        nl_dict = {'sine': Sine(), 'relu': nn.ReLU(inplace=True), 'softplus': nn.Softplus(beta=100),
                    'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}
        nl = nl_dict[nonlinearity]

        self.net = []
        self.net.append(MetaSequential(BatchLinear(in_features, hidden_features), nl))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(BatchLinear(hidden_features, hidden_features), nl))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features), nl))

        self.net = MetaSequential(*self.net)

        if init_type == 'siren':
            self.net.apply(sine_init)
            self.net[0].apply(first_layer_sine_init)

        elif init_type == 'geometric_sine':
            self.net.apply(geom_sine_init)
            self.net[0].apply(first_layer_geom_sine_init)
            self.net[-2].apply(second_last_layer_geom_sine_init)
            self.net[-1].apply(last_layer_geom_sine_init)

        elif init_type == 'mfgi':
            self.net.apply(geom_sine_init)
            self.net[0].apply(first_layer_mfgi_init)
            self.net[1].apply(second_layer_mfgi_init)
            self.net[-2].apply(second_last_layer_geom_sine_init)
            self.net[-1].apply(last_layer_geom_sine_init)

        elif init_type == 'geometric_relu':
            self.net.apply(geom_relu_init)
            self.net[-1].apply(geom_relu_last_layers_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))

        if self.init_type == 'mfgi' or self.init_type == 'geometric_sine':
            radius, scaling = self.sphere_init_params
            output = torch.sign(output)*torch.sqrt(output.abs()+1e-8)
            output -= radius # 1.6
            output *= scaling # 1.0

        return output



class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def forward(self, input):
        # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


################################# SIREN's initialization ###################################
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See SIREN paper supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

################################# sine geometric initialization ###################################

def geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_output = m.weight.size(0)
            m.weight.uniform_(-np.sqrt(3 / num_output), np.sqrt(3 / num_output))
            m.bias.uniform_(-1 / (num_output * 1000), 1 / (num_output * 1000))
            m.weight.data /= 30
            m.bias.data /= 30

def first_layer_geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_output = m.weight.size(0)
            m.weight.uniform_(-np.sqrt(3 / num_output), np.sqrt(3 / num_output))
            m.bias.uniform_(-1 / (num_output * 1000), 1 / (num_output * 1000))
            m.weight.data /= 30
            m.bias.data /= 30


def second_last_layer_geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_output = m.weight.size(0)
            assert m.weight.shape == (num_output, num_output)
            m.weight.data = 0.5 * np.pi * torch.eye(num_output) + 0.001 * torch.randn(num_output, num_output)
            m.bias.data = 0.5 * np.pi * torch.ones(num_output, ) + 0.001 * torch.randn(num_output)
            m.weight.data /= 30
            m.bias.data /= 30

def last_layer_geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            assert m.weight.shape == (1, num_input)
            assert m.bias.shape == (1,)
            # m.weight.data = -1 * torch.ones(1, num_input) + 0.001 * torch.randn(num_input)
            m.weight.data = -1 * torch.ones(1, num_input) + 0.00001 * torch.randn(num_input)
            m.bias.data = torch.zeros(1) + num_input


################################# multi frequency geometric initialization ###################################
periods = [1, 30] # Number of periods of sine the values of each section of the output vector should hit
# periods = [1, 60] # Number of periods of sine the values of each section of the output vector should hit
portion_per_period = np.array([0.25, 0.75]) # Portion of values per section/period

def first_layer_mfgi_init(m):
    global periods
    global portion_per_period
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            num_output = m.weight.size(0)
            num_per_period = (portion_per_period * num_output).astype(int) # Number of values per section/period
            assert len(periods) == len(num_per_period)
            assert sum(num_per_period) == num_output
            weights = []
            for i in range(0, len(periods)):
                period = periods[i]
                num = num_per_period[i]
                scale = 30/period
                weights.append(torch.zeros(num,num_input).uniform_(-np.sqrt(3 / num_input) / scale, np.sqrt(3 / num_input) / scale))
            W0_new = torch.cat(weights, axis=0)
            m.weight.data = W0_new

def second_layer_mfgi_init(m):
    global portion_per_period
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            assert m.weight.shape == (num_input, num_input)
            num_per_period = (portion_per_period * num_input).astype(int) # Number of values per section/period
            k = num_per_period[0] # the portion that only hits the first period
            # W1_new = torch.zeros(num_input, num_input).uniform_(-np.sqrt(3 / num_input), np.sqrt(3 / num_input) / 30) * 0.00001
            W1_new = torch.zeros(num_input, num_input).uniform_(-np.sqrt(3 / num_input), np.sqrt(3 / num_input) / 30) * 0.0005
            W1_new_1 = torch.zeros(k, k).uniform_(-np.sqrt(3 / num_input) / 30, np.sqrt(3 / num_input) / 30)
            W1_new[:k, :k] = W1_new_1
            m.weight.data = W1_new

################################# geometric initialization used in SAL and IGR ###################################
def geom_relu_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            out_dims = m.out_features

            m.weight.normal_(mean=0.0, std=np.sqrt(2) / np.sqrt(out_dims))
            m.bias.data = torch.zeros_like(m.bias.data)

def geom_relu_last_layers_init(m):
    radius_init = 1
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.normal_(mean=np.sqrt(np.pi) / np.sqrt(num_input), std=0.00001)
            m.bias.data = torch.Tensor([-radius_init])


