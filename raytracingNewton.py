import numpy as np
from classesNewton import  Illum, Sphere, Newton, normalize, Light, Ray
import matplotlib.pyplot as plt


width = 200
height = 200

#pixels = np.empty(shape=(height, width, 3))
pixels = np.zeros(shape=(height, width, 3))

ratio = float(width) / height

cam_pos = [0, 0, 1]
camera = np.array(cam_pos)

light_src_pos = np.array([5,3,1])
light = Light(light_src_pos, Illum(np.array([0.6,0.6,0.6]), np.array([1,1,1]), np.array([1,1,1])))


width_pos = [-1, 1] #minimum and maximum width
height_pos = [-ratio, ratio] #minimum and maximum height


#pos : np.array, radius, color : np.array, illum : Illum, shininess, reflection : float, camera: np.array, lineDirection: np.array):
functions = [Sphere(np.array([-0.5, 0, -1]), 0.7, np.array([57/256, 68/256, 188/256]), Illum(np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1])), 80, 0.4, camera),
            Sphere(np.array([0.2, -0.3, -2]), 0.7, np.array([123/256, 6/256, 35/256]), Illum(np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1])), 80, 0.4, camera),
            Sphere(np.array([0, -1000, 0]), 999, np.array([150/256, 70/256, 150/256]), Illum(np.array([0.1, 0.1, 0.1]), np.array([0.6, 0.6, 0.6]), np.array([1,1,1])), 80, 0.1, camera)]

for i, y in enumerate(np.linspace(height_pos[0], height_pos[1], height)):
    for j, x in enumerate(np.linspace(width_pos[0], width_pos[1], width)):
        
        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)
        cumulative_reflection = 1
        
        for bounce in range(5):
            
            #Ray is called with: number of Iterations, stepsize, the coordinates of the camera, the direction, a list of spheres, the Newton method, light, cumulative_reflection
            #Returns, the color as np.array, the coordinates of the better point calculated with the Newton method, obj_reflection, if there has been an intersection reflection=1 otherwise is 0, id of the sphere which was hit
            colorToBePlaced, betterPoint, obj_reflection, intersection, idFunction = Ray(250, 0.01, camera, direction, functions, Newton, light, cumulative_reflection)
            if intersection == 1:
                cumulative_reflection *= obj_reflection
                normal = normalize(betterPoint - functions[idFunction].center())
                direction = direction - 2 * np.dot(direction, normal) * normal
                origin = betterPoint
                pixels[height-i-1, j-1] += colorToBePlaced
            else:
                pixels[height-i-1, j-1] += np.array([0,0,0])
                break


#Temporarily fixing when some valuse are bigger than 1
for i in range(height):
    for j in range(width):
        for k in range(3):
            if pixels[i][j][k] >1:
                pixels[i][j][k]=0.99

plt.imsave('Q1.png', pixels)