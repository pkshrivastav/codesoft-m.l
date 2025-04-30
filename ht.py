# handwritten_text_rnn.py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import os
import gzip
import shutil
import urllib.request
import string

# Download EMNIST dataset
def download_and_extract_emnist():
    if not os.path.exists("emnist-balanced-train-images-idx3-ubyte.gz"):
        print("Downloading EMNIST dataset...")
        base_url = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
        os.system(f"curl -O https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip")
        os.system("tar -xf gzip.zip")

# Load EMNIST (Balanced) images and labels
def load_emnist_images(file):
    with gzip.open(file, 'rb') as f:
        f.read(16)  # Skip header
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8).reshape(-1, 28, 28)
        return data

def load_emnist_labels(file):
    with gzip.open(file, 'rb') as f:
        f.read(8)  # Skip header
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels

# Data preprocessing
def preprocess_images(images):
    return images / 255.0

def decode_labels_to_chars(labels):
    # Basic mapping of balanced EMNIST to characters (subset)
    chars = string.digits + string.ascii_lowercase
    return ''.join([chars[l % len(chars)] for l in labels])

# Create sequences for character-level modeling
def create_sequences(text, seq_length=40):
    inputs, targets = [], []
    for i in range(len(text) - seq_length):
        inputs.append(text[i:i+seq_length])
        targets.append(text[i+seq_length])
    return inputs, targets

# Vectorization
def vectorize_sequences(inputs, targets, char2idx):
    x = np.zeros((len(inputs), len(inputs[0])), dtype=np.int32)
    y = np.zeros(len(targets), dtype=np.int32)
    for i, seq in enumerate(inputs):
        x[i] = [char2idx[c] for c in seq]
        y[i] = char2idx[targets[i]]
    return x, y

# Build character-level RNN model
def build_model(vocab_size, seq_length):
    model = tf.keras.Sequential([
        layers.Embedding(vocab_size, 64, input_length=seq_length),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(128),
        layers.Dense(64, activation='relu'),
        layers.Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

# Generate new text
def generate_text(model, start_string, char2idx, idx2char, num_generate=300, temperature=1.0):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    result = []

    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = predictions[:, -1, :] / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.concat([input_eval[:, 1:], tf.expand_dims([predicted_id], 0)], axis=1)
        result.append(idx2char[predicted_id])
    
    return start_string + ''.join(result)

# Plot sample images
def plot_sample_images(images, labels, num=25):
    plt.figure(figsize=(10, 10))
    for i in range(num):
        plt.subplot(5, 5, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(labels[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# =============================
#         MAIN SCRIPT
# =============================

# 1. Download and load data
print("Loading EMNIST data...")
download_and_extract_emnist()

# Manually downloaded dataset (place files in same dir if above fails)
img_path = "C:/Users/princ/python/mlp/p...jpg"
#lbl_path = "./emnist-balanced-train-labels-idx1-ubyte.gz"
lbl_path = "C:/Users/princ/python/mlp/p...jpg"
images = load_emnist_images(img_path)
labels = load_emnist_labels(lbl_path)

images = preprocess_images(images)
characters = decode_labels_to_chars(labels)

# 2. Create sequences
seq_length = 40
inputs, targets = create_sequences(characters, seq_length)

# 3. Vectorize
vocab = sorted(set(characters))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
x, y = vectorize_sequences(inputs, targets, char2idx)

# 4. Split data
split = int(0.8 * len(x))
x_train, y_train = x[:split], y[:split]
x_val, y_val = x[split:], y[split:]

# 5. Build and train model
model = build_model(len(vocab), seq_length)
print(model.summary())

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128)

# 6. Generate text
print("\nGenerated Handwritten-like Text:")
print(generate_text(model, start_string="hello ", char2idx=char2idx, idx2char=idx2char))

# 7. Plot sample images
plot_sample_images(images, characters)

# 8. Visualize training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
