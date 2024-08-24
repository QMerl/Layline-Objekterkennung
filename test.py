"""
Datei um unser trainiertes Modell anzuwenden und denn Servo zu steuern
Finale Datei

"""

import pyfirmata
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torch.utils.data import DataLoader
from model import Yolov1
from utils import (
    non_max_suppression,
    cellboxes_to_boxes,
    load_checkpoint,
)
from utilstest import plotxservo
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
WEIGHT_DECAY = 0
LOAD_MODEL = True
LOAD_MODEL_FILE = "overfit.pth.tar"



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def main():

    model = Yolov1(split_size=7, num_boxes=2, num_classes=1).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
    
    
    #aktiviert die Kamera und teilt ihr eine Variabel zu
    cap = cv2.VideoCapture(1)
    board = pyfirmata.Arduino('COM6')
    it = pyfirmata.util.Iterator(board)
    it.start()
    servo_pin = board.get_pin('d:9:s')
    servo_pin2 = board.get_pin('d:8:s')
    angley=90
    servo_pin.write(angley)
    anglex=90
    servo_pin2.write(anglex)
    # Starte einen Iterator, um die serielle Verbindung zu verwalten
    
    
    while True:
        
        #Bild wird aufgenommen und der Variabel Frame zugeteilt 
        ret, frame = cap.read()
        #Bild wird für die KI bearbeitet aber separat abgespeichert damit nachher das unbearbeitete Bild gezeigt werden kann
        image = cv2.resize(frame,(448,448))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #wird in die richtige Array grösse bearbeitet
        result = image.transpose(2, 0, 1)
        result = np.tile(result, (16, 1, 1, 1))
        #Die Zahlen werden zwischen 0 und 1 scaliert 
        result = result/255
        
        #wird in ein Torch.Tensor umgewandelt
        x = torch.from_numpy(result).cuda().float()
        
        #Wird zur GPU geschickt
        x = x.to(DEVICE)
            
       
        #berechnet die BBoxes und welche 
        bboxes = cellboxes_to_boxes(model(x))
        #Es werden die BBoxes die sich überlappen zusammengeführt und nur die mit dem besten Ergebniss wird behalten
        bboxes = non_max_suppression(bboxes[1], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        
        angley, anglex = plotxservo(frame, bboxes,angley, anglex,servo_pin,servo_pin2)
        
        cv2.waitKey(1) 

    board.exit()


if __name__ == "__main__":
    main()