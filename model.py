
# This code is the implementation of the DiGS model and loss functions
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

class AnthropometryEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[64, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class Conv1DRefiner(nn.Module):
    def __init__(self, embed_dim, num_layers=3, kernel_size=3):
        super().__init__()
        # Ensure padding is set to keep the sequence length the same
        padding = (kernel_size - 1) // 2

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim,
                                    kernel_size=kernel_size, padding=padding))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm([embed_dim, 92])) # Norm over feature and sequence dims

        self.refiner = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim) -> (N, 92, 32)

        # Conv1d expects (batch, channels, seq_len), so we permute the dimensions
        x = x.permute(0, 2, 1) # -> (N, 32, 92)

        refined_x = self.refiner(x)

        # Permute back to the original dimension order for the final head
        refined_x = refined_x.permute(0, 2, 1) # -> (N, 92, 32)
        return refined_x

class SelfAttentionBlock(nn.Module):
    """ A standard Transformer self-attention block. """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # Note: batch_first=True is crucial for our tensor shapes
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(), # GELU is often used in modern transformers
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, embed_dim)
        # Self-attention: query, key, and value are all `x`
        attn_output, _ = self.attention(x, x, x)
        # First residual connection
        x = self.norm1(x + attn_output)
        # Second residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x



class CrossAttentionEncoder(nn.Module):

    def __init__(self, query_dim, context_dim, latent_dim, num_heads=8):
        super().__init__()
        # Ensure latent_dim is divisible by num_heads for MultiheadAttention
        if latent_dim % num_heads != 0:
            raise ValueError(f"'latent_dim' ({latent_dim}) must be divisible by 'num_heads' ({num_heads}).")

        self.embed_dim = latent_dim  # The core dimension for the attention mechanism

        # The multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True  # We use (batch, seq, feature) format
        )

        # Linear layers to project input and context to the attention's embedding dimension
        self.query_proj = nn.Linear(query_dim, self.embed_dim)
        self.context_proj = nn.Linear(context_dim, self.embed_dim)

        # A standard feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim)
        )

        # Layer normalization for stabilizing training
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, query_in, context_in):
        """
        Forward pass for the cross-attention encoder.
        - query_in: The primary input tensor (query). Shape: (batch_size, query_dim)
        - context_in: The context tensor. Shape: (batch_size, context_dim)
        """
        # Reshape vectors to sequences of length 1
        if query_in.dim() == 2:
            query_seq = query_in.unsqueeze(1)
        if context_in.dim() == 2:
            context_seq = context_in.unsqueeze(1)

        # Project inputs to the embedding dimension
        query = self.query_proj(query_seq)
        key = self.context_proj(context_seq)
        value = self.context_proj(context_seq)

        # Apply cross-attention
        attn_output, _ = self.attention(query=query, key=key, value=value)

        # First residual connection and normalization
        x_res = self.norm1(query + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(x_res)

        # Second residual connection and normalization
        processed_output = self.norm2(x_res + ffn_output)

        # Squeeze the sequence dimension to get the final vector
        output = processed_output.squeeze(1)
        return output


class CrossAttentionBlock(nn.Module):
    # This is essentially your original module's core logic
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        # query: (batch, 1, embed_dim)
        # key, value: (batch, 1, embed_dim)
        attn_output, _ = self.attention(query, key, value)
        x_res = self.norm1(query + attn_output)
        ffn_output = self.ffn(x_res)
        processed_output = self.norm2(x_res + ffn_output)
        return processed_output



class StackedCrossAttentionEncoder(nn.Module):
    def __init__(self, query_dim, context_dim, latent_dim, num_heads=8, num_layers=4):
        super().__init__()
        self.embed_dim = latent_dim
        # Project inputs once at the beginning
        self.query_proj = nn.Linear(query_dim, self.embed_dim)
        self.context_proj = nn.Linear(context_dim, self.embed_dim)

        # Create a list of attention blocks
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(self.embed_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, query_in, context_in):
        # Reshape and project inputs
        query = self.query_proj(query_in.unsqueeze(1))
        context_seq = context_in.unsqueeze(1)
        key = self.context_proj(context_seq)
        value = self.context_proj(context_seq) # Or a separate projection for value

        # Pass through all layers, updating the query each time
        for layer in self.layers:
            query = layer(query, key, value)

        # Squeeze the sequence dimension
        return query.squeeze(1)

class HRTFNetwork(nn.Module):
    # In HRTFNetwork class's __init__ method:

    def __init__(self, anthropometry_dim, target_freq_dim, # Added target_freq_dim
                latent_dim=32, num_heads=8, decoder_hidden_dim=128,
                decoder_n_hidden_layers=4, init_type='siren', nonlinearity='sine'): # Removed target_freq_dim from here
        super().__init__()
        #self.encoder = AnthropometryEncoder(anthropometry_dim, latent_dim)
        query_dim = 2
        self.encoder = StackedCrossAttentionEncoder(
            query_dim=query_dim,            # Input from az_el
            context_dim=anthropometry_dim,  # Input from anthropometry data
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=4
        )

        # Input to decoder: 2 (preprocessed az, el) + latent_dim
        # Output features: Number of frequency bins to predict
        self.decoder = FCBlock(
            in_features=2 + latent_dim,
            out_features=target_freq_dim, # Set output size to number of frequencies
            num_hidden_layers=decoder_n_hidden_layers,
            hidden_features=decoder_hidden_dim,
            outermost_linear=True,
            nonlinearity=nonlinearity,
            init_type=init_type
        )

    def forward(self, az_el, anthropometry):
        # az_el: (N, 2) -- azimuth and elevation
        # anthropometry: (N, anthropometry_dim)
        #latent = self.encoder(anthropometry)  # (N, latent_dim)
        latent = self.encoder(query_in=az_el, context_in=anthropometry)
        x = torch.cat([az_el, latent], dim=-1)  # (N, 2 + latent_dim)
        return self.decoder(x)


class HRTFNetwork_self(nn.Module):
    def __init__(self,
                 anthropometry_dim,
                 target_freq_dim=92,
                 latent_dim=64,
                 num_heads=8,
                 # --- NEW parameters for frequency attention ---
                 freq_embed_dim=32,      # Feature dimension for each frequency
                 freq_attn_layers=3,     # How many self-attention blocks to stack
                 # --------------------------------------------
                 decoder_hidden_dim=256,
                 decoder_n_hidden_layers=4,
                 init_type='siren',
                 nonlinearity='sine'):
        super().__init__()

        # --- Store key dimensions for reshaping later ---
        self.target_freq_dim = target_freq_dim
        self.freq_embed_dim = freq_embed_dim
        # ----------------------------------------------

        # 1. ENCODER (no changes here)
        # This part still produces a single latent vector conditioned on space and anthropometry.
        query_dim = 2 # az_el
        self.encoder = StackedCrossAttentionEncoder(
            query_dim=query_dim,
            context_dim=anthropometry_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=4
        )

        # 2. INITIAL DECODER (repurposed)
        # Instead of predicting final magnitudes, this MLP predicts the "raw" feature vectors
        # for all frequencies at once.
        self.initial_decoder = FCBlock(
            in_features=2 + latent_dim, # Input is still conditioned latent code + original az_el
            # NEW: Output enough features for all frequencies
            out_features=target_freq_dim * freq_embed_dim,
            num_hidden_layers=decoder_n_hidden_layers,
            hidden_features=decoder_hidden_dim,
            outermost_linear=True,
            nonlinearity=nonlinearity,
            init_type=init_type
        )

        # 3. FREQUENCY ATTENTION REFINER (new)
        # This is a stack of self-attention blocks to process the sequence of frequencies.
        self.frequency_refiner = nn.Sequential(
            *[SelfAttentionBlock(freq_embed_dim, num_heads) for _ in range(freq_attn_layers)]
        )

        # 4. FINAL HEAD (new)
        # This final small MLP projects each refined frequency feature vector to a single magnitude value.
        self.final_head = nn.Sequential(
            nn.Linear(freq_embed_dim, freq_embed_dim // 2),
            nn.GELU(),
            nn.Linear(freq_embed_dim // 2, 1)
            # We don't add a final ReLU here, as it's often better to add it
            # outside the model or use a loss function robust to raw outputs (logits).
        )


    def forward(self, az_el, anthropometry):
        # az_el: (N, 2)
        # anthropometry: (N, anthropometry_dim)
        batch_size = az_el.shape[0]

        # --- 1. Encoder ---
        # Get the latent vector conditioned on spatial position and subject anthropometry.
        latent = self.encoder(query_in=az_el, context_in=anthropometry) # Shape: (N, latent_dim)

        # --- 2. Initial Decoder ---
        # Prepare input for the decoder and generate raw frequency features.
        x = torch.cat([az_el, latent], dim=-1) # Shape: (N, 2 + latent_dim)
        raw_freq_features = self.initial_decoder(x) # Shape: (N, target_freq_dim * freq_embed_dim)

        # --- 3. Reshape and Refine ---
        # Reshape the flat output into a sequence for the attention module.
        # This is the key step: we now treat frequencies as a sequence.
        freq_sequence = raw_freq_features.view(
            batch_size,
            self.target_freq_dim,
            self.freq_embed_dim
        ) # Shape: (N, 92, 32)

        # Pass the sequence through the self-attention blocks.
        refined_sequence = self.frequency_refiner(freq_sequence) # Shape: (N, 92, 32)

        # --- 4. Final Projection ---
        # Project each refined frequency vector to its final magnitude.
        magnitudes = self.final_head(refined_sequence) # Shape: (N, 92, 1)

        # Squeeze the last dimension to get the desired output shape.
        return magnitudes.squeeze(-1) # Final Shape: (N, 92)


class HRTFNetwork_conv(nn.Module):
    def __init__(self,
                 anthropometry_dim,
                 target_freq_dim=92,
                 latent_dim=64,
                 num_heads=8,
                 # --- NEW parameters for frequency attention ---
                 freq_embed_dim=32,
                 # --------------------------------------------
                 decoder_hidden_dim=256,
                 decoder_n_hidden_layers=4,
                 init_type='siren',
                 nonlinearity='sine'):
        super().__init__()

        # --- Store key dimensions for reshaping later ---
        self.target_freq_dim = target_freq_dim
        self.freq_embed_dim = freq_embed_dim
        # ----------------------------------------------

        # 1. ENCODER (no changes here)
        # This part still produces a single latent vector conditioned on space and anthropometry.
        query_dim = 2 # az_el
        self.encoder = StackedCrossAttentionEncoder(
            query_dim=query_dim,
            context_dim=anthropometry_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=4
        )

        # 2. INITIAL DECODER (repurposed)
        # Instead of predicting final magnitudes, this MLP predicts the "raw" feature vectors
        # for all frequencies at once.
        self.initial_decoder = FCBlock(
            in_features=2 + latent_dim, # Input is still conditioned latent code + original az_el
            # NEW: Output enough features for all frequencies
            out_features=target_freq_dim * freq_embed_dim,
            num_hidden_layers=decoder_n_hidden_layers,
            hidden_features=decoder_hidden_dim,
            outermost_linear=True,
            nonlinearity=nonlinearity,
            init_type=init_type
        )


        self.frequency_refiner = Conv1DRefiner(embed_dim=freq_embed_dim, num_layers=3, kernel_size=3)

        # 4. FINAL HEAD (new)
        # This final small MLP projects each refined frequency feature vector to a single magnitude value.
        self.final_head = nn.Sequential(
            nn.Linear(freq_embed_dim, freq_embed_dim // 2),
            nn.GELU(),
            nn.Linear(freq_embed_dim // 2, 1)
            # We don't add a final ReLU here, as it's often better to add it
            # outside the model or use a loss function robust to raw outputs (logits).
        )


    def forward(self, az_el, anthropometry):
        # az_el: (N, 2)
        # anthropometry: (N, anthropometry_dim)
        batch_size = az_el.shape[0]

        # --- 1. Encoder ---
        # Get the latent vector conditioned on spatial position and subject anthropometry.
        latent = self.encoder(query_in=az_el, context_in=anthropometry) # Shape: (N, latent_dim)

        # --- 2. Initial Decoder ---
        # Prepare input for the decoder and generate raw frequency features.
        x = torch.cat([az_el, latent], dim=-1) # Shape: (N, 2 + latent_dim)
        raw_freq_features = self.initial_decoder(x) # Shape: (N, target_freq_dim * freq_embed_dim)

        # --- 3. Reshape and Refine ---
        # Reshape the flat output into a sequence for the attention module.
        # This is the key step: we now treat frequencies as a sequence.
        freq_sequence = raw_freq_features.view(
            batch_size,
            self.target_freq_dim,
            self.freq_embed_dim
        ) # Shape: (N, 92, 32)

        # Pass the sequence through the self-attention blocks.
        refined_sequence = self.frequency_refiner(freq_sequence) # Shape: (N, 92, 32)

        # --- 4. Final Projection ---
        # Project each refined frequency vector to its final magnitude.
        magnitudes = self.final_head(refined_sequence) # Shape: (N, 92, 1)

        # Squeeze the last dimension to get the desired output shape.
        return magnitudes.squeeze(-1) # Final Shape: (N, 92)




class HRTFNetwork3D(nn.Module):
    # In HRTFNetwork class's __init__ method:

    def __init__(self, anthropometry_dim, target_freq_dim, # Added target_freq_dim
                latent_dim=32, num_heads=8, decoder_hidden_dim=128,
                decoder_n_hidden_layers=4, init_type='siren', nonlinearity='sine'): # Removed target_freq_dim from here
        super().__init__()
        #self.encoder = AnthropometryEncoder(anthropometry_dim, latent_dim)
        query_dim = 3
        self.encoder = StackedCrossAttentionEncoder(
            query_dim=query_dim,            # Input from az_el
            context_dim=anthropometry_dim,  # Input from anthropometry data
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=4
        )

        # Input to decoder: 2 (preprocessed az, el) + latent_dim
        # Output features: Number of frequency bins to predict
        self.decoder = FCBlock(
            in_features=3 + latent_dim,
            out_features=target_freq_dim, # Set output size to number of frequencies
            num_hidden_layers=decoder_n_hidden_layers,
            hidden_features=decoder_hidden_dim,
            outermost_linear=True,
            nonlinearity=nonlinearity,
            init_type=init_type
        )

    def forward(self, az_el, anthropometry):
        # az_el: (N, 2) -- azimuth and elevation
        # anthropometry: (N, anthropometry_dim)
        #latent = self.encoder(anthropometry)  # (N, latent_dim)
        latent = self.encoder(query_in=az_el, context_in=anthropometry)
        x = torch.cat([az_el, latent], dim=-1)  # (N, 3 + latent_dim)
        return self.decoder(x)

class HRTFNetwork_film(nn.Module):
    """
    HRTF Network conditioned on anthropometry using FiLM layers.
    """
    def __init__(self, anthropometry_dim, target_freq_dim, latent_dim,
                 decoder_hidden_dim=128, decoder_n_hidden_layers=4,
                 init_type='siren', nonlinearity='sine'):
        super().__init__()
        self.decoder_hidden_dim = decoder_hidden_dim

        # The number of layers we need to modulate is the input layer + all hidden layers
        self.num_modulated_layers = decoder_n_hidden_layers + 1

        # 1. The main network processes the azimuth and elevation
        self.main_network = FiLMedFCBlock(
            in_features=2,  # Input is (azimuth, elevation)
            out_features=target_freq_dim,
            num_hidden_layers=decoder_n_hidden_layers,
            hidden_features=decoder_hidden_dim,
            outermost_linear=True,
            nonlinearity=nonlinearity,
            init_type=init_type
        )

        # 2. The FiLM generator processes the anthropometric data
        self.film_generator = FiLMGenerator(
            cond_dim=anthropometry_dim,
            num_modulated_layers=self.num_modulated_layers,
            film_hidden_dim=decoder_hidden_dim
        )

    def forward(self, az_el, anthropometry):
        # az_el: (N, 2)
        # anthropometry: (N, anthropometry_dim)

        # Step 1: Generate FiLM parameters from the anthropometry data.
        film_params = self.film_generator(anthropometry)

        # Step 2: Reshape the parameters to separate gammas and betas.
        # The shape becomes (N, num_layers, hidden_dim, 2)
        film_params = film_params.view(
            anthropometry.shape[0],
            self.num_modulated_layers,
            self.decoder_hidden_dim,
            2
        )
        gammas = film_params[..., 0]  # Shape: (N, num_layers, hidden_dim)
        betas = film_params[..., 1]   # Shape: (N, num_layers, hidden_dim)

        # Step 3: Pass coordinates and FiLM params to the main network to get the final output.
        output = self.main_network(az_el, gammas, betas)
        return output


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

class FiLMedFCBlock(nn.Module):
    """
    A Fully Connected (FC) block that is modulated by FiLM parameters.
    This network processes the primary input (e.g., coordinates).
    """
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', init_type='siren'):
        super().__init__()

        nl_dict = {'sine': Sine(), 'relu': nn.ReLU(inplace=True)}
        self.nl = nl_dict.get(nonlinearity, Sine()) # Default to Sine

        self.net = nn.ModuleList()

        # Input layer
        self.net.append(nn.Linear(in_features, hidden_features))

        # Hidden layers
        for _ in range(num_hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))

        # Output layer
        self.net.append(nn.Linear(hidden_features, out_features))

        self.outermost_linear = outermost_linear

        # Apply weight initialization
        if init_type == 'siren':
            self.apply(sine_init)
            self.net[0].apply(first_layer_sine_init)
        # Add other init types here if needed

    def forward(self, x, gammas, betas):
        """
        Forward pass with FiLM modulation.
        - x: The primary input tensor (e.g., coordinates).
        - gammas: Scaling parameters from the FiLMGenerator.
        - betas: Shifting parameters from the FiLMGenerator.
        """
        # Modulate the input layer and all hidden layers
        for i, layer in enumerate(self.net[:-1]):
            x = layer(x)
            # Apply FiLM: y = gamma * x + beta
            # Unsqueeze is for broadcasting across the feature dimension
            x = gammas[:, i, :] * x + betas[:, i, :]
            x = self.nl(x)

        # Pass through the final layer
        x = self.net[-1](x)
        if not self.outermost_linear:
            x = self.nl(x)

        return x


# New module to generate the FiLM parameters
class FiLMGenerator(nn.Module):
    """
    Generates FiLM parameters (gamma and beta) from a conditioning input.
    """
    def __init__(self, cond_dim, num_modulated_layers, film_hidden_dim):
        super().__init__()

        # The generator must produce 2 values (gamma, beta) for each feature
        # in each modulated layer of the main network.
        output_size = num_modulated_layers * film_hidden_dim * 2

        # A simple MLP to generate the parameters
        self.generator = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

        # Initialize the final layer to output values close to an identity
        # transformation (gamma=1, beta=0) at the start of training.
        with torch.no_grad():
            self.generator[-1].weight.fill_(0.)
            # Split the bias tensor to initialize gammas and betas separately
            gamma_bias, beta_bias = torch.chunk(self.generator[-1].bias, 2, dim=0)
            gamma_bias.fill_(1.)
            beta_bias.fill_(0.)

    def forward(self, cond_input):
        return self.generator(cond_input)





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