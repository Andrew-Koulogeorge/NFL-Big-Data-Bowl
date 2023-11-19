# Video Vision Transformer Model Implementation v2

# __Author:__ Jack Friedman <br>
# __Date:__ 11/17/2024 <br>
# __Adapted from:__ Aritra Roy Gosthipaty and Ayush Thakur (https://github.com/keras-team/keras-io/blob/master/examples/vision/vivit.py) <br>
# __Original Paper:__ ViViT: A Video Vision Transformer (https://arxiv.org/abs/2103.15691) by Arnab et al. <br>
# __Updates from v1:__ 
# - Added synthetic data
# - Multichannel implementation

## Import libraries
import os
import io
import time 
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, callbacks
from keras import backend as K
from keras.saving import register_keras_serializable
from keras.models import load_model, save_model
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../preprocessing')
from Preprocessing_v4 import *
from DataLoader import load_data
# Setting seed for reproducibility
SEED = 42
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
keras.utils.set_random_seed(SEED)


## Step 0: Define key hyperparameters and constants
"""
Code authors tuned these hyperparameters. 

We will use the same (or similar) ones for v0
"""

# DATA
BATCH_SIZE = 64
AUTO = tf.data.AUTOTUNE
FRAMES_PER_PLAY = 12
INPUT_SHAPE = (FRAMES_PER_PLAY, 120, 54, 10)
SYNTHETIC_DATA_PROPORTION = 0.5

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 90

# TUBELET EMBEDDING
PATCH_SIZE = (6, 6, 6)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8


## Step 1: Load and preprocess data
print("LOADING AND PREPROCESSING DATA")

### Step 1A: Load data
# Load data
[games_df, players_df, plays_df, tracking_df] = load_data()

### Step 1B: Preprocess data
# Preprocess data
tracking_df_clean = preprocess_all_df(plays_df, games_df, players_df, tracking_df)

### Step 1C: Get tensors
print("GETTING TENSORS")
# Get tensors
start_time = time.time()
tensor_list, labels = prepare_4d_tensors(tracking_df_clean, min_frames= 12)
print("processing time:", timedelta(seconds = (time.time() - start_time)))
### Step 1D: One-hot encode labels
# STEP 0: Get min and max labels (see See www.kaggle.com/competitions/nfl-big-data-bowl-2020/overview/evaluation)
indexed_labels = [round(label) + 99 for label in labels]
min_idx_y = np.min(indexed_labels)
max_idx_y = np.max(indexed_labels)
print('min yardIndex:', min_idx_y)
print('max yardIndex:', max_idx_y)

# STEP 1: CALCULATE NUMBER OF CLASSES (YARDS)
num_classes_y = max_idx_y - min_idx_y + 1
print('num classes:', num_classes_y)

# Ensure min_idx_y is the same type as label_indexed
min_idx_y = tf.cast(min_idx_y, tf.int32)
num_classes_y = tf.cast(num_classes_y, tf.int32)

# STEP 2: CONVERT LABELS TO OHE 
labels_ohe = []
for label in labels:
    # Index each label
    label_indexed = tf.cast(tf.round(label), tf.int32) + 99  # Indexing the label

    # One-hot encode the label
    label_one_hot = tf.one_hot(label_indexed - min_idx_y, depth=num_classes_y)
    
    labels_ohe += [label_one_hot]

## Step 2: Create Synthetic Data
# Function that creaets a synthetic observation from 2 data points
def mixup_observations(x1, y1, x2, y2, alpha=0.27):
    # Get parameter lambda
    l = np.random.beta(a = alpha, b = alpha)

    # Get new x and y combos
    x_new = tf.multiply(x1, l) + tf.multiply(x2, 1 - l)
    y_new = tf.multiply(y1, l) + tf.multiply(y2, 1 - l)

    # Clip 0-1
    x_new = tf.clip_by_value(x_new, 0, 1)
    y_new = tf.clip_by_value(y_new, 0, 1)
    
    return x_new, y_new

def generate_synthetic_data(X, y, num_points):
    X_synthetic = []
    y_synthetic = []
    for _ in range(num_points):
        # Pick 2 random observations
        i = int(np.random.uniform(low = 0, high = len(X)))
        j = int(np.random.uniform(low = 0, high = len(X)))

        # Create a synthetic observation
        x_new, y_new = mixup_observations(X[i], y[i], X[j], y[j], alpha=0.27)

        X_synthetic += [x_new]
        y_synthetic += [y_new]

    return X_synthetic, y_synthetic

# Calcualte number of synthetic datapointst o generate
num_points = int(SYNTHETIC_DATA_PROPORTION * len(tensor_list) / (1 - SYNTHETIC_DATA_PROPORTION))

# Get synthetic data
print("generating synthetic data...")
synthetic_tensors, synthetic_labels = generate_synthetic_data(tensor_list, labels_ohe, num_points)
print("data points created:", len(synthetic_labels))

# Add synthetic data
full_tensor_list = tensor_list + synthetic_tensors
full_label_list = labels_ohe + synthetic_labels
print("proportion synthetic data:", len(synthetic_labels) / (len(full_label_list)))
print("new total observations:", len(full_label_list))
## Step 3: Prep data for training (model-specific preprocessing)
### Step 3A: Train-test split
X_train, X_test, y_train, y_test = train_test_split(tensor_list, labels, test_size=0.2, random_state=SEED)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=SEED) # 0.25 x 0.8 = 0.2
### Step 3B: Build dataloaders
@tf.function
def preprocess(frames: tf.Tensor, label: tf.Tensor, num_classes_y: int = num_classes_y, min_idx_y: int = min_idx_y):
    """Preprocess the frames tensors and parse the labels."""
    # Preprocess images
    frames = tf.image.convert_image_dtype(
        frames[
            ..., tf.newaxis
        ],  # The new axis is to help for further processing with Conv3D layers
        tf.float32,
    )

    # Parse label
    label = tf.cast(label, tf.float32)
    
    return frames, label


def prepare_dataloader(
    videos: np.ndarray,
    labels: np.ndarray,
    loader_type: str = "train",
    batch_size: int = BATCH_SIZE,
):
    """Utility function to prepare the dataloader."""
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

    if loader_type == "train":
        dataset = dataset.shuffle(BATCH_SIZE * 2)

    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataloader

trainloader = prepare_dataloader(X_train, y_train, "train")
validloader = prepare_dataloader(X_val, y_val, "valid")
testloader = prepare_dataloader(X_test, y_test, "test")
### Step 3C: Define classes for building model
#### (i) Tublet Embedding & Positional Encoding
"""
## Tubelet Embedding

In ViTs, an image is divided into patches, which are then spatially
flattened, a process known as tokenization. For a video, one can
repeat this process for individual frames. **Uniform frame sampling**
as suggested by the authors is a tokenization scheme in which we
sample frames from the video clip and perform simple ViT tokenization.

| ![uniform frame sampling](https://i.imgur.com/aaPyLPX.png) |
| :--: |
| Uniform Frame Sampling [Source](https://arxiv.org/abs/2103.15691) |

**Tubelet Embedding** is different in terms of capturing temporal
information from the video.
First, we extract volumes from the video -- these volumes contain
patches of the frame and the temporal information as well. The volumes
are then flattened to build video tokens.

| ![tubelet embedding](https://i.imgur.com/9G7QTfV.png) |
| :--: |
| Tubelet Embedding [Source](https://arxiv.org/abs/2103.15691) |
"""

# @register_keras_serializable
class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches
"""
## Positional Embedding

This layer adds positional information to the encoded video tokens.
"""

# @register_keras_serializable
class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens
#### (ii) Model architecture 
"""
## Video Vision Transformer

The authors suggest 4 variants of Vision Transformer:

- Spatio-temporal attention
- Factorized encoder
- Factorized self-attention
- Factorized dot-product attention

In this example, we will implement the **Spatio-temporal attention**
model for simplicity. The following code snippet is heavily inspired from
[Image classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer/).
One can also refer to the
[official repository of ViViT](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit)
which contains all the variants, implemented in JAX.
"""

# Loss function - Continuous Ranked Probability Score
def crps(y_true, y_pred):
    loss = K.mean(K.sum((K.cumsum(y_pred, axis = 1) - K.cumsum(y_true, axis=1))**2, axis=1))/199
    return loss

def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    input_shape=INPUT_SHAPE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    output_size=num_classes_y,
    l1_reg=0.01,  # L1 regularization factor
    l2_reg=0.01,  # L2 regularization factor
    pdrop=0.3    # Base dropout probability for stochastic depth
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = tubelet_embedder(inputs)
    # Encode patches.
    encoded_patches = positional_encoder(patches)

    # Create multiple layers of the Transformer block.
    for l in range(transformer_layers):
        # Compute dropout probabiltiy via stoachstic depth regularization
        depth_drop_prob = l / transformer_layers * pdrop

        # Layer normalization and MHSA
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=depth_drop_prob
        )(x1, x1)

        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization and MLP
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = keras.Sequential(
            [
                layers.Dense(units=embed_dim * 4, 
                             activation=tf.nn.gelu,
                             kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
                layers.Dense(units=embed_dim, 
                             activation=tf.nn.gelu,
                             kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
            ]
        )(x3)

        # Skip connection
        encoded_patches = layers.Add()([x3, x2])

    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)

    # Classify outputs.
    outputs = layers.Dense(units=output_size, activation="softmax")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
## Step 4: Train Model
"""
## Train
"""


def run_experiment():
    # Initialize model
    model = create_vivit_classifier(
        tubelet_embedder=TubeletEmbedding(
            embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
        ),
        positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
    )

    # Compile the model with the optimizer, loss function
    # and the metrics.
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=crps
    )
    # To implement early stopping, use Keras callbacks when fitting the model
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    # Train the model
    print("training now...")
    start_time = time.time()
    # Train the model
    history = model.fit(
        trainloader,
        epochs=EPOCHS,
        validation_data=validloader,
        callbacks=[early_stopping]  # Include the custom Metric here
    )
    print("finished training")
    print("training time:", timedelta(seconds = (time.time() - start_time)))

    # Print metrics
    train_crps = model.evaluate(trainloader)
    print(f"Train CRPS: {round(train_crps, 4)}")
    
    val_crps = model.evaluate(validloader)
    print(f"Val CRPS: {round(val_crps, 4)}")

    test_crps = model.evaluate(testloader)
    print(f"Test CRPS: {round(test_crps, 4)}")

    return model


model = run_experiment()
# Save model
model.save('vivit_v2_model.h5')
# Print metrics
train_crps = model.predict(trainloader)
print(f"Train CRPS: {round(train_crps, 4)}")

val_crps = model.evaluate(validloader)
print(f"Val CRPS: {round(val_crps, 4)}")

test_crps = model.evaluate(testloader)
print(f"Test CRPS: {round(test_crps, 4)}")
# # Load model
# model_pickled = load_model('vivit_v1_model.h5', custom_objects={'TubeletEmbedding': TubeletEmbedding, 
#                                                                 'PositionalEncoder': PositionalEncoder})