from data_loader import SiameseDataGenerator
from model import build_base_cnn, visualize_embeddings, evaluate_model
from utilities import display_img, euclidean_distance, contrastive_loss, pair_accuracy

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.utils import plot_model

def main():
    # run_tasks() # to run exercise tasks

    # Run AI model
    full_df = pd.read_csv("signatures-dataset/filtered_dataframe_binarized.csv")
    
    unique_persons = full_df['person_id'].unique()
    # 15% seperate test set.
    train_val_persons, test_persons = train_test_split(
        unique_persons, test_size=0.15, random_state=42
    )
    
    # 70% training 15% validation
    train_persons, val_persons = train_test_split(
        train_val_persons, test_size=0.176, random_state=42 # 0.176 av 0.85 er ca 15%
    )    

    train_df = full_df[full_df['person_id'].isin(train_persons)]
    val_df = full_df[full_df['person_id'].isin(val_persons)]
    test_df = full_df[full_df['person_id'].isin(test_persons)]
    
    train_gen = SiameseDataGenerator(train_df, batch_size=32)
    val_gen = SiameseDataGenerator(val_df, batch_size=32)
    test_gen = SiameseDataGenerator(test_df, batch_size=32)
    
    # Input shape for the model.
    INPUT_SHAPE = (375, 616, 1)

    img1_input = layers.Input(shape=INPUT_SHAPE, name="signature_1")
    img2_input = layers.Input(shape=INPUT_SHAPE, name="signature_2")

    # Get the base network for embedding extraction
    base_network = build_base_cnn(input_shape=INPUT_SHAPE)

    # get an image of the model
    # plot_model(
    #     base_network, 
    #     to_file='siamese_architecture.png', 
    #     show_shapes=True, 
    #     show_layer_names=True,
    #     expand_nested=True # Dette gjør at den bretter ut base_cnn så du ser alle Conv2D-lagene
    # )
    # return 


    # Send both inputs through the same base network (shared weights)
    emb1 = base_network(img1_input)
    emb2 = base_network(img2_input)

    # Calculate the euclidean distance between the two embeddings
    distance = layers.Lambda(euclidean_distance, name="distance")([emb1, emb2])

    # Build and compile the Siames Network
    siamese_model = Model(inputs=[img1_input, img2_input], outputs=distance)
    siamese_model.compile(optimizer='adam', loss=contrastive_loss, metrics=[pair_accuracy])
    
    siamese_model.summary()
    
    print("Generating t-SNE plot before training...")
    visualize_embeddings(base_network, test_gen, title="t-SNE before training (random initialization)")

    early_stopper = EarlyStopping(
        monitor='val_pair_accuracy', 
        patience=8,                  
        mode='max',                  
        restore_best_weights=True    
    )
    
    history = siamese_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=16,
        callbacks=[early_stopper]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))
    
    # Plot for Loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    # ax1.set_title('Contrastive Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot for Accuracy
    ax2.plot(history.history['pair_accuracy'], label='Training-accuracy')
    ax2.plot(history.history['val_pair_accuracy'], label='Validation-accuracy')
    # ax2.set_title('Accuracy (Threshold=0.5)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    fig.savefig("figures/contrastive_loss_and_accuracy.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Evaluation
    print("Evaluerer modell...")
    evaluate_model(siamese_model, test_gen)

    # t-SNE Visualization
    print("Generating t-SNE after training...")
    visualize_embeddings(base_network, test_gen, title="t-SNE after training")
    pass


if __name__ == "__main__":
    main()
