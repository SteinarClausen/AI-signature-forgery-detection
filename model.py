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
    
    # Convolution + Pooling
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten and Embedding
    x = layers.Flatten()(x)
    
    embeddings = layers.Dense(embedding_dim, activation=None)(x)
    
    return Model(inputs, embeddings, name="base_cnn")



def evaluate_model(siamese_model, test_gen, threshold=0.5):
    y_true = []
    y_pred_dist = []

    for i in range(len(test_gen)):
        (img1, img2), labels = test_gen[i]
        distances = siamese_model.predict([img1, img2], verbose=0)
        
        y_true.extend(labels.flatten())
        y_pred_dist.extend(distances.flatten())

    y_true = np.array(y_true)
    y_pred_dist = np.array(y_pred_dist)

    # Distance < threshold means "Real" (0), if not "False" (1)
    y_pred = (y_pred_dist > threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    # plt.title(f"Confusion Matrix (Threshold={threshold})")
    plt.savefig("figures/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    return cm



def visualize_embeddings(base_network, gen, num_batches=5, title="t-SNE of Signature Embeddings"):
    all_embeddings = []
    all_labels = []

    for i in range(num_batches):
        (img1, img2), labels = gen[i]
        # Pick out embedding for img1.
        embeddings = base_network.predict(img1, verbose=0)
        all_embeddings.append(embeddings)
        all_labels.append(labels)

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.vstack(all_labels).flatten()

    # Reduce dimensions to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(5, 4))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter, ticks=[0, 1], label='0: Real, 1: Fake')
    # plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"figures/{title}")
    plt.show()