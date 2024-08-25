"""
Kreiert ein Pytorch Datenset um die Bilder und Labels (Datenset) zu laden
"""

import torch
import os
import pandas as pd
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=1, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # Konvertiert die bboxen zu den respektieven Zellen
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j representiert die Zell Reihe und Spalte
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            """
            Berechnen der Höhe und Breite der bboxen relativ zur 
            entsprechenden Zelle
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)

            Um die Breite relativ zur Zelle zu finden:
            width_pixels/cell_pixels, Dies führt zur aufgelisteten Formel.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )
            # Wenn noch keine Box in Zelle i,j
            # Das heisst eine Zelle kann nur eine bbox haben
            if label_matrix[i, j, 1] == 0:
                # Setze das ein Objekt existiert
                label_matrix[i, j, 1] = 1
                # Box Koordinaten
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 2:6] = box_coordinates
                # Setze die Wahrscheinlichkeit für Class Propability
                label_matrix[i, j, class_label] = 1
        return image, label_matrix