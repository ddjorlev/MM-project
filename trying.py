from classesNewton import Sphere, Illum
import numpy as np

camera = np.array([1, 2, 3])
functions = [Sphere(np.array([-0.5, 0, -1]), 0.7, np.array([57/256, 68/256, 188/256]), Illum(np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1])), 80, 0.4, camera),
            Sphere(np.array([0.2, -0.3, -2]), 0.7, np.array([123/256, 6/256, 35/256]), Illum(np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1])), 80, 0.4, camera),
            Sphere(np.array([0, -1000, 0]), 999, np.array([150/256, 70/256, 150/256]), Illum(np.array([0.1, 0.1, 0.1]), np.array([0.6, 0.6, 0.6]), np.array([1,1,1])), 80, 0.1, camera)]

numberOfSpheres = len(functions)

prevSign = [0] * numberOfSpheres

for i in range(numberOfSpheres):
    prevSign[i]=functions[i].pos

for i in range(numberOfSpheres):
    print(prevSign[i])