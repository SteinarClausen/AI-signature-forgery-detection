from task2 import read_image
from task3 import show_image_histogram
from task5 import task5
from task6 import task6
from task7 import task7
from task8 import task8a, task8b 

from data_loader import SiameseDataGenerator
from model import build_base_cnn, visualize_embeddings, evaluate_model
from utilities import display_img, euclidean_distance, contrastive_loss

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

from tensorflow.keras import layers, Model

def main():
    image_str = "exampleSignature.png"
    # Task 2: read and print the image
    # print(read_image(image_str))
    
    # Task 3: Show two image histograms
    # show_image_histogram(image_str, bins=64)
    # show_image_histogram(image_str, bins=256)

    # Task 5: Apply the convolution kernals to the images
    # img_A, img_B, img_C, img_D = task5(image_str)

    # display_img(img_A)
    # display_img(img_B)
    # display_img(img_C)
    # display_img(img_D)

    # task 6: threshold the image and display the resulting image
    # task6(image_str)
    # Use this one for the signature correcting.
    # task 7: optimize the rotation of the image to minimize the height of the bounding box
    # image_str_diag = "exampleSignature_diag.png"
    # task7(image_str_diag)
    # task 8
    # task8a(image_str)
    # task8b(image_str)

    # Dataloader class
    # 1. Last inn dataframen én gang
    full_df = pd.read_csv("signatures-dataset/filtered_dataframe_binarized.csv")
    
    # 2. Hent unike skribenter og splitt dem (80% trening, 20% test)
    unique_persons = full_df['person_id'].unique()
    # 2. Første splitt: Ta ut 15% til et helt separat TEST-sett
    train_val_persons, test_persons = train_test_split(
        unique_persons, test_size=0.15, random_state=42
    )
    
    # 3. Andre splitt: Del resten i Trening (ca 70% totalt) og Validering (ca 15% totalt)
    train_persons, val_persons = train_test_split(
        train_val_persons, test_size=0.176, random_state=42 # 0.176 av 0.85 er ca 15%
    )    
    # 3. Lag undersett av dataframen
    train_df = full_df[full_df['person_id'].isin(train_persons)]
    val_df = full_df[full_df['person_id'].isin(val_persons)]
    test_df = full_df[full_df['person_id'].isin(test_persons)]
    
    # 4. Oppdater generatorene til å ta inn DataFrame (se endring i klasse under)
    train_gen = SiameseDataGenerator(train_df, batch_size=16)
    val_gen = SiameseDataGenerator(val_df, batch_size=16)
    test_gen = SiameseDataGenerator(test_df, batch_size=16)

    # print(siam_data_generator[0])  # Get the first batch of data
    
    # Input shape for the model.
    INPUT_SHAPE = (375, 616, 1)

    img1_input = layers.Input(shape=INPUT_SHAPE, name="signature_1")
    img2_input = layers.Input(shape=INPUT_SHAPE, name="signature_2")

    # 2. Hent base-nettverket
    base_network = build_base_cnn(input_shape=INPUT_SHAPE)

    # 3. Send BEGGE bildene gjennom SAMME nettverk (Deler parametere)
    emb1 = base_network(img1_input)
    emb2 = base_network(img2_input)

    # 4. Regn ut Euklidsk avstand mellom embeddingene
    distance = layers.Lambda(euclidean_distance, name="distance")([emb1, emb2])

    # 5. Bygg og kompiler den ferdige Siamese-modellen
    siamese_model = Model(inputs=[img1_input, img2_input], outputs=distance)
    siamese_model.compile(optimizer='adam', loss=contrastive_loss)

    siamese_model.summary()
    
    history = siamese_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10
    )

    plt.plot(history.history['loss'], label='Trenings-tap')
    plt.plot(history.history['val_loss'], label='Validerings-tap')
    plt.title('Modellens læringskurve')
    plt.xlabel('Epoker')
    plt.ylabel('Tap (Contrastive Loss)')
    plt.legend()
    plt.show()

    # 12c: Evaluering
    print("Evaluerer modell...")
    evaluate_model(siamese_model, test_gen)

    # 12d: t-SNE Visualisering
    print("Genererer t-SNE plott...")
    visualize_embeddings(base_network, test_gen)


    pass


if __name__ == "__main__":
    main()
