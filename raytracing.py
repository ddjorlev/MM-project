import numpy as np
from classes import normalize
import matplotlib.pyplot as plt
from classes import Ray, Sphere, Illum, Light
import math

cam_pos = [0,0,1]
camera = np.array(cam_pos)
#3,0,1
light_src_pos = np.array([5,3,1])
light = Light(light_src_pos, Illum(np.array([0.6,0.6,0.6]), np.array([1,1,1]), np.array([1,1,1])))

width = 800
height = 800

#camera is at (0,0,1) and we want screen to be centered at (0,0,0)
ratio = height/width

width_pos = [-1, 1] #minimum and maximum width
height_pos = [-ratio, ratio] #minimum and maximum height
#from the top 2 lines we can say screen is from (-1, -ratio, 0) to (1, ratio, 0)
#this is because we want increasing width or height to increase detail(pixel density)

pixels = np.empty(shape = (width, height, 3))
spheres = [Sphere(np.array([-0.5, 0, -1]), 0.7, np.array([57/256, 68/256, 188/256]), Illum(np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1])), 80, 0.4),
            Sphere(np.array([0.2, -0.3, -2]), 0.7, np.array([123/256, 6/256, 35/256]), Illum(np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1])), 80, 0.4),
            Sphere(np.array([0, -1000, 0]), 999, np.array([150/256, 70/256, 150/256]), Illum(np.array([0.1, 0.1, 0.1]), np.array([0.6, 0.6, 0.6]), np.array([1,1,1])), 80, 0.1)]

max_bounce = 4
for i, y in enumerate(np.linspace(height_pos[0], height_pos[1], height)[1:]):
    for j, x in enumerate(np.linspace(width_pos[0], width_pos[1], width)[1:]):
        new_pixel = np.array([x,y,0])
        #np.append(pixels, new_pixel)
        start = camera
        line_direction = new_pixel - start
        line_direction = line_direction/np.linalg.norm(line_direction)
        ray = Ray(new_pixel, line_direction, np.array([0,0,0]))
        cumulative_reflection = 1
        shade = 0
        for k in range(max_bounce):
            t_min = math.inf #closest intersection
            min_sphere = None
            for sphere in spheres:
                t = ray.sphere_intersect(sphere)
                if(t and t < t_min):
                    t_min = t
                    min_sphere = sphere
            obj_color = sphere.color
            if t_min != math.inf:
                cumulative_reflection = ray.set_color_cosine_bounce(new_pixel + t_min*ray.vector, min_sphere, light, cam_pos, cumulative_reflection)

                if k == 1:
                    bounce_pos = ray.pos
                    line_to_src = light_src_pos - bounce_pos
                    new_ray = Ray(bounce_pos, line_to_src, np.array([0,0,0]))
                    new_intersection = math.inf #closest intersection
                    intersect_sphere = None
                    for sphere in spheres:
                        t = new_ray.sphere_intersect(sphere)
                        if(t and t < new_intersection):
                            new_intersection = t
                            intersect_sphere = sphere
                    if new_intersection != math.inf:
                        shade = 1 
                    
                ray.bounce(new_pixel + t_min*ray.vector, min_sphere) 
                    
            # start = new_pixel + t_min*ray.vector
            # line_direction = bounce(line_direction, )
            # line_direction = line_direction/np.linalg.norm(line_direction)
            if (shade != 0 and k == 2):
                ray.color = ray.color * 0.7

            pixels[width - i - 1, j - 1] = ray.color

plt.imsave('test2.png', pixels)
        