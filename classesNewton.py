import numpy as np


def normalize(vector: np.array):
    return vector / np.linalg.norm(vector)

class Illum:
    def __init__(self, ambient : np.array, diffuse : np.array, specular : np.array):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular

class Light:
    def __init__(self, pos : np.array, illum : Illum):
        self.pos = pos
        self.illum = illum

class Sphere:
    def __init__(self, pos : np.array, radius, color : np.array, illum : Illum, shininess, reflection : float, camera: np.array):
        #pos = [x,y,z]
        self.pos = pos
        self.illum = illum
        self.radius = radius
        self.color = color
        self.shininess = shininess
        self.reflection = reflection
        self.camera=camera

    #Function for calculating the value of a sphere in the point x[x1, y1, z1]
    def calculateF(self, x: np.array):
        return (x[0] - self.pos[0])**2 + (x[1] - self.pos[1])**2 + (x[2] - self.pos[2])**2 - self.radius**2
    
    #Function for calculating the first derrivative of a sphere in the point t[x, y, z]
    def dx(self, t:np.array) :
        return 2*(t[0] - self.pos[0])
    
    #Function for calculating the second derrivative of a sphere in the point t[x, y, z]
    def dy(self, t:np.array) :
        return 2*(t[1] - self.pos[1])
    
    #Function for calculating the third derrivative of a sphere in the point t[x, y, z]
    def dz(self, t:np.array) :
        return 2*(t[2] - self.pos[2])
    
    #Function g(t):=f(x0 + a*t, y0 + a*t, z0 + a*t)
    #used to find the exact point of intersection
    def g(self, t:np.array, lineDirection:np.array) :
        newX=self.camera[0]+t*lineDirection[0]
        newY=self.camera[1]+t*lineDirection[1]
        newZ=self.camera[2]+t*lineDirection[2]
                                                 
        return self.calculateF(np.array([newX, newY, newZ]))
    
    #Derrivative of the function above
    def gdot(self, t:np.array, lineDirection:np.array) :
        return (
            lineDirection[0] * self.dx(np.array([self.camera[0] +t*lineDirection[0], self.camera[1]+t*lineDirection[1], self.camera[2]+t*lineDirection[2]])) +
            lineDirection[1] * self.dy(np.array([self.camera[0] +t*lineDirection[0], self.camera[1]+t*lineDirection[1], self.camera[2]+t*lineDirection[2]])) +
            lineDirection[2] * self.dz(np.array([self.camera[0] +t*lineDirection[0], self.camera[1]+t*lineDirection[1], self.camera[2]+t*lineDirection[2]]))
            )
    
    #function for calculating the sign in the point t[x, y, z]
    def sign(self, t:np.array):
        s = np.sign(self.calculateF(t))
        if s == 0:
            s=1
        return s
    
    #Returning the center of the sphere
    def center(self):
        c = np.array([self.pos[0], self.pos[1], self.pos[2]])
        return c


def Newton(s: Sphere, lineDirection: np.array, x0:float ,epsilon,max_iter):
    xn = x0
    for n in range(0,max_iter):
        fxn = s.g(xn, lineDirection)
        if abs(fxn) < epsilon:
            return xn
        Dfxn = s.gdot(xn, lineDirection)
        if Dfxn == 0: 
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    
    return xn


def Ray(itMax: int, stepsize: int, camera: np.array, lineDirection:np.array, sphere: Sphere, newton: Newton, light: Light, cumulative_reflection):
   
    it=0
    t=stepsize

    stepDirection = stepsize * lineDirection

    prevPoint = camera

    newPoint= prevPoint + stepsize

    prevSign=sphere.sign(camera)
    currSign=sphere.sign(newPoint)

    intersection=0

    while(it< itMax):
        
        it=it+1
        t=t+stepsize

        prevPoint = newPoint
        newPoint=newPoint + stepDirection

        prevSign=currSign
        currSign = sphere.sign(newPoint)
        
        newtonPoint=0
        
        betterPoint=newPoint
        
        intersection=0
        if currSign != prevSign:
            
            startOfApproximation = (t + (t-stepsize) ) /2

            newtonPoint = newton(sphere, lineDirection ,startOfApproximation, 1e-10, 100)
            betterPoint = camera + newtonPoint*lineDirection
            intersection =1
            break
        
    black = np.array([0.0, 0.0, 0.0])
    colorToAdd = np.array([0.0, 0.0, 0.0])
   
    colorToAdd += sphere.illum.ambient * light.illum.ambient
    
    centerOfTheSphere=sphere.center()

    normal = normalize(betterPoint - centerOfTheSphere)
    intersection_to_light = normalize(light.pos - betterPoint)
    
    colorToAdd += sphere.illum.diffuse * light.illum.diffuse * np.dot(intersection_to_light,normal)
    
    intersection_to_cam = camera - betterPoint

    cam_to_light = normalize(betterPoint + intersection_to_cam)
    colorToAdd += sphere.illum.specular * light.illum.specular * np.dot(normal, cam_to_light) ** (sphere.shininess/4)
    
    if intersection ==1 :
        return sphere.color * np.clip(colorToAdd, 0, 1) * cumulative_reflection, betterPoint, sphere.reflection
    else:
        return black, np.array([np.inf, np.inf, np.inf]), 1
