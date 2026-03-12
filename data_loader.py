import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from task7 import get_crop_dimensions, minimize_height_finite_difference, rotate_image
from utilities import get_image_binary

import matplotlib.pyplot as plt

class SiameseDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=16):
        if isinstance(data, str):
            self.df = pd.read_csv(csv_file)
        else:
            self.df = data
        
        self.batch_size = batch_size
        
        # Grupper etter person for å gjøre det enkelt å trekke tilfeldige par
        self.persons = self.df['person_id'].unique()
        
        # Lag ordbøker (dictionaries) som holder lister med filstier for hver person
        self.reals = {p: self.df[(self.df['person_id'] == p) & (self.df['real'] == True)]['image_path_filtered'].tolist() for p in self.persons}
        self.forgeries = {p: self.df[(self.df['person_id'] == p) & (self.df['real'] == False)]['image_path_filtered'].tolist() for p in self.persons}

    def __len__(self):
        return self.df.shape[0] // self.batch_size 

    def _load_image(self, path):
        # Laster inn som gråtone, legger til kanal-dimensjon og normaliserer [0, 1]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=-1) / 255.0
        return img

    def __getitem__(self, index):
        batch_s1, batch_s2, labels = [], [], []
        
        for _ in range(self.batch_size):
            # 1. Velg en tilfeldig skribent
            person = np.random.choice(self.persons)
            
            # 2. Kast mynt: 0 for ekte-ekte, 1 for ekte-forfalsket
            is_forgery = np.random.choice([0, 1])
            
            real_list = self.reals[person]
            forg_list = self.forgeries[person]
            
            # Sørg for at personen faktisk har minst 2 ekte signaturer før vi lager et ekte-ekte par
            if is_forgery == 0 and len(real_list) >= 2:
                img1_path, img2_path = np.random.choice(real_list, 2, replace=False)
                label = 0.0
            else:
                img1_path = np.random.choice(real_list)
                img2_path = np.random.choice(forg_list)
                label = 1.0
                
            batch_s1.append(self._load_image(img1_path))
            batch_s2.append(self._load_image(img2_path))
            labels.append(label)
            
        return (np.array(batch_s1), np.array(batch_s2)), np.array(labels)


    def image_sizes(self):
        """Returns: x_sizes, y_sizes, and plots them"""
        x_sizes: list[int] = []
        y_sizes: list[int] = []
        for idx, row in self.df.iterrows():
            img_path = row["image_path_filtered"]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            binary_img = get_image_binary(img)
            theta_opt, h_opt, hist = minimize_height_finite_difference(binary_img, theta0=0.0, h0=1.0, max_iter=120)

            rot_opt = rotate_image(binary_img, theta_opt)
            y0, y1, x0, x1 = get_crop_dimensions(rot_opt)

            x_sizes.append(x1 - x0)
            y_sizes.append(y1 - y0)
        
        plt.scatter(x_sizes, y_sizes)
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.title("Scatter plot of signature image sizes")
        plt.grid()
        plt.show()
        return x_sizes, y_sizes