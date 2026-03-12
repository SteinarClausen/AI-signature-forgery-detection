import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE

CANVAS_SIZE = (375, 616)
INPUT_SHAPE = (375, 616, 1)


def build_base_cnn(input_shape=INPUT_SHAPE, embedding_dim=128):
    inputs = layers.Input(shape=input_shape)
    
    # 1. Convolution + Pooling
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    # Tensor shape: (None, 148, 298, 32)
    x = layers.MaxPooling2D((2, 2))(x)
    # Tensor shape: (None, 74, 149, 32)
    
    # 2. Convolution + Pooling
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    # Tensor shape: (None, 72, 147, 64)
    x = layers.MaxPooling2D((2, 2))(x)
    # Tensor shape: (None, 36, 73, 64)
    
    # 3. Convolution + Pooling
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    # Tensor shape: (None, 34, 71, 128)
    x = layers.MaxPooling2D((2, 2))(x)
    # Tensor shape: (None, 17, 35, 128)
    
    # Flatten og Embedding
    x = layers.Flatten()(x)
    # Tensor shape: (None, 76160)
    
    # Dense layer for å lage selve embedding-vektoren e i R^d
    embeddings = layers.Dense(embedding_dim, activation=None)(x)
    # Tensor shape: (None, 128)  <- Dette er d!
    
    return Model(inputs, embeddings, name="base_cnn")



def evaluate_model(siamese_model, test_gen, threshold=0.5):
    y_true = []
    y_pred_dist = []

    # Gå gjennom test-settet (her kan du bruke val_gen eller en egen test_gen)
    for i in range(len(test_gen)):
        (img1, img2), labels = test_gen[i]
        distances = siamese_model.predict([img1, img2], verbose=0)
        
        y_true.extend(labels.flatten())
        y_pred_dist.extend(distances.flatten())

    y_true = np.array(y_true)
    y_pred_dist = np.array(y_pred_dist)

    # Konverter avstand til binær prediksjon: 
    # Avstand < threshold betyr "Ekte" (0), ellers "Falsk" (1)
    y_pred = (y_pred_dist > threshold).astype(int)

    # Lag og vis forvirringsmatrisen
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ekte', 'Falsk'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Threshold={threshold})")
    plt.show()

    return cm



def visualize_embeddings(base_network, gen, num_batches=5):
    all_embeddings = []
    all_labels = []

    for i in range(num_batches):
        (img1, img2), labels = gen[i]
        # Vi trekker ut embeddinger for det første bildet i hvert par
        embeddings = base_network.predict(img1, verbose=0)
        all_embeddings.append(embeddings)
        all_labels.append(labels)

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.vstack(all_labels).flatten()

    # Reduser fra f.eks. 128 dimensjoner til 2 for plotting
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, ticks=[0, 1], label='0: Ekte, 1: Falsk')
    plt.title("t-SNE av Signature Embeddings")
    plt.show()