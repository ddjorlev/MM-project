import numpy as np

class InvalidDimension(Exception):
    "Invalid number of arguments of pos."
    pass

class InvalidArrayDimensionPos(Exception):
    "Invalid dimensionality of pos"
    pass

class InvalidArrayDimensionVector(Exception):
    "Invalid dimensionality of vector"
    pass

class Sphere:
    def __init__(self, pos : np.array, radius : int):
        if(len(pos) != 3):
            raise InvalidDimension()
        if(len(pos.shape) != 1):
            raise InvalidArrayDimensionPos()
        if(type(radius) != int):
            raise TypeError()
        self.pos = pos
        self.radius = radius

class Ray:
    #TODO: add color to ray
    def __init__(self, pos : np.array, vector : np.array):
        if(len(pos) != 3 or len(vector) != 3):
            raise InvalidDimension()
        if(len(pos.shape) != 1):
            raise InvalidArrayDimensionPos()
        if(len(vector.shape) != 1):
            raise InvalidArrayDimensionVector()
        self.pos = np.array(pos)
        self.vector = vector/np.linalg.norm(vector)
    
    def sphere_intersect(self, sphere : Sphere) -> int:
        a = 1 #(self.vector[0]**2 + self.vector[1]**2 + self.vector[2]**2) length of line vector will be always 1
        #print("a",a)
        m = self.pos[0] - sphere.pos[0]
        #print("m",m)
        n = self.pos[1] - sphere.pos[1]
        #print("n",n)
        l = self.pos[2] - sphere.pos[2]
        #print("l",l)
        b = 2*m*self.vector[0] + 2*n*self.vector[1] + 2*l*self.vector[2]
        #print("b",b)
        c = m**2 + n**2 + l**2 - sphere.radius**2
        #print("c",c)
        delta = b**2 - 4*a*c
        #print("delta",delta)
        if(delta > 0): #this means that we actually intersect
            t1 = (-b + np.sqrt(delta))/(2*a)
            t2 = (-b - np.sqrt(delta))/(2*a)
            #print("t1",t1)
            #print("t2",t2)
            if(t1 > 0 and t2 > 0):
                return min(t1,t2) #when we do intersect we intersect in 2 spots, we want the smaller one because that is the one we are "seeing"
        return None