import numpy as np
<<<<<<< HEAD
=======
import matplotlib.pyplot as plt
from classes import Ray, Sphere

cam_pos = [0,0,1]
camera = np.array(cam_pos)

width = 400
height = 800

#camera is at (0,0,1) and we want screen to be centered at (0,0,0)
ratio = height/width

width_pos = [-1, 1] #minimum and maximum width
height_pos = [-ratio, ratio] #minimum and maximum height
#from the top 2 lines we can say screen is from (-1, -ratio, 0) to (1, ratio, 0)
#this is because we want increasing width or height to increase detail(pixel density)

pixels = np.array([])
for y in np.linspace(height_pos[0], height_pos[1], height):
    for x in np.linspace(width_pos[0], width_pos[1], width):
        new_pixel = [x,y,0]
        np.append(pixels, new_pixel)

        line_direction = new_pixel - camera
        line_direction = line_direction/np.linalg.norm(line_direction)


