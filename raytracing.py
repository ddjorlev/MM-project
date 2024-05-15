import numpy as np

import matplotlib.pyplot as plt
from classes import Ray, Sphere, Illum, Light
import math

cam_pos = [0,0,1]
camera = np.array(cam_pos)

light_src_pos = np.array([5,3,1])
light = Light(light_src_pos, Illum(np.array([0.6,0.6,0.6]), np.array([1,1,1]), np.array([1,1,1])))

width = 50
height = 50

#camera is at (0,0,1) and we want screen to be centered at (0,0,0)
ratio = height/width

width_pos = [-1, 1] #minimum and maximum width
height_pos = [-ratio, ratio] #minimum and maximum height
#from the top 2 lines we can say screen is from (-1, -ratio, 0) to (1, ratio, 0)
#this is because we want increasing width or height to increase detail(pixel density)

pixels = np.empty(shape = (width, height, 3))
spheres = [Sphere(np.array([-0.2, 0, -1]), 0.7, np.array([57/256, 68/256, 188/256]), Illum(np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1])), 100),
            Sphere(np.array([0.5, -0.3, -2]), 0.7, np.array([123/256, 6/256, 35/256]), Illum(np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1])), 100)]

for i, y in enumerate(np.linspace(height_pos[0], height_pos[1], height)[1:]):
    for j, x in enumerate(np.linspace(width_pos[0], width_pos[1], width)[1:]):
        new_pixel = np.array([x,y,0])
        np.append(pixels, new_pixel)

        # for k in range(3):
        line_direction = new_pixel - camera
        line_direction = line_direction/np.linalg.norm(line_direction)
        ray = Ray(new_pixel, line_direction, np.array([0,0,0]))
        t_min = math.inf #closest intersection
        min_sphere = None
        for sphere in spheres:
            t = ray.sphere_intersect(sphere)
            if(t and t < t_min):
                t_min = t
                min_sphere = sphere
        obj_color = sphere.color
        if t_min != math.inf:
            ray.set_color_cosine_2(new_pixel + t_min*ray.vector, min_sphere, light, cam_pos)    
            pixels[width - i, j] = ray.color

plt.imsave('sphere.png', pixels)
        



