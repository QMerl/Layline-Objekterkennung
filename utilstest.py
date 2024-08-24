import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import cv2
import pyfirmata
import time
import math




def plotxservo(image, boxes,angley, anglex,servo_pin, servo_pin2):
    """
    Zeigt Bild mit bbox und bewegt die zwei Serven in die entsprechende Richtung
    Der Winkel zum Ausgang wird auch ausgegeben
    """
    

    im = image
    #Höhe und Breite des Bildes wird festgelegt
    height, width, _ = im.shape
    
    #Es werden die Variabeln iniziert zur Berechnung zur Distanz der bboxen zur Mitte in Pixeln
    x_zu_mitte=0
    y_zu_mitte=0
    ix=()
    iy=()
    iXY = ()
    #Bounding Box und Mittelpunkt wird aufgezeichnet 
    #durchläuft die einzelnen Listen also Boxen wenn es im Bild zwei gibt sind es zwei Listen 
    for box in boxes:
        #Es werden die Mittelpunkt Koordinaten der Box und die Breite und Länge herausgeschrieben
        # box[0] is x midpoint, box[2] is width
        # box[1] is y midpoint, box[3] is height
        box = box[2:]
        box = np.array(box)
        
        
        
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        #Es wird die obere Linke Ecke definiert
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        
        #Die zwei Variabeln oben werden in ein tuple geschrieben 
        rectrco= (int(upper_left_x * width), int(upper_left_y * height))
        
        #Die Linke untere Ecke der BBox wird definiert
        bottom_l_x=box[0]+ box[2]/2
        bottom_l_y=box[1]+ box[3]/2
        #werden in ein Tuple geschrieben 
        rectlco = (int(bottom_l_x*width), int(bottom_l_y*height))

        
        x_zu_mitte = box[0]*width - width / 2
        y_zu_mitte = box[1]*width - height / 2

        ix = ix + (x_zu_mitte,)
        iy = iy + (y_zu_mitte,)

        zwischenXY = math.sqrt(x_zu_mitte * x_zu_mitte + y_zu_mitte *y_zu_mitte)
        iXY = iXY + (zwischenXY,)
        
        #Wird geplotet und dabei ist wichtig das alles tupples sind un Intetcher 
        if rectlco is not None:
           
            cv2.rectangle(im, rectrco, rectlco, (0, 255, 0), 2)
            cv2.circle(im, (int(box[0]*width), int(box[1]*height)), 2, (0, 255, 0), 2)

    length = len(iXY)
    if length > 0:
        min_position = 0
        min_value = 0
        
        for i in range(1, len(iXY)):
            if abs(iXY[i]) < min_value:
                min_value = iXY[i]
                min_position = i
        if iy[min_position] < 0:
            if iy[min_position] < -150:
                angley = angley + 10
            elif iy[min_position] < -40:
                angley = angley + 5
            elif iy[min_position] > -10:
                angley = angley + 1
            
                
        if iy[min_position] > 0:
            if iy[min_position] > 150:
                angley = angley - 10
            elif iy[min_position] > 40:
                angley = angley - 5
            elif iy[min_position] < 10:
                angley = angley - 1

        if angley>180:
            angley = 180
        if angley<0:
            angley = 0

        if ix[min_position] < 0:
            if ix[min_position] < -150:
                anglex = anglex + 10
            elif ix[min_position] < -40:
                anglex = anglex + 5
            elif ix[min_position] > -10:
                anglex = anglex + 1

        if ix[min_position] > 0:
            if ix[min_position] > 150:
                anglex = anglex - 10
            elif ix[min_position] > 40:
                anglex = anglex - 5
            elif ix[min_position] < 10:
                anglex = anglex - 1

        if anglex>180:
            anglex = 180
        if anglex<0:
            anglex = 0
        
        servo_pin.write(angley)
        servo_pin2.write(anglex)

        print(anglex-90)
    
        
    #zeigt das Image
    cv2.imshow('Object Detection', im)

    return angley, anglex