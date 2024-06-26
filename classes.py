import numpy as np
import numbers

def normalize(vec):
    return vec/np.linalg.norm(vec)

class InvalidDimension(Exception):
    "Invalid number of arguments of pos."
    pass

class InvalidArrayDimensionPos(Exception):
    "Invalid dimensionality of pos"
    pass

class InvalidArrayDimensionVector(Exception):
    "Invalid dimensionality of vector"
    pass

class InvalidArrayDimensionColor(Exception):
    "Invalid dimensionality of color"
    pass

class Illum:
    def __init__(self, ambient : np.array, diffuse : np.array, specular : np.array):
        if(len(ambient) != 3):
            raise InvalidDimension()
        if(len(ambient.shape) != 1):
            raise InvalidArrayDimensionPos()
        if(len(diffuse) != 3):
            raise InvalidDimension()
        if(len(diffuse.shape) != 1):
            raise InvalidArrayDimensionPos()
        if(len(specular) != 3):
            raise InvalidDimension()
        if(len(specular.shape) != 1):
            raise InvalidArrayDimensionPos()
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular

class Sphere:
    def __init__(self, pos : np.array, radius, color : np.array, illum : Illum, shininess, reflection : float):
        if(len(pos) != 3):
            raise InvalidDimension()
        if(len(pos.shape) != 1):
            raise InvalidArrayDimensionPos()
        if(len(color) != 3):
            raise InvalidDimension()
        if(len(color.shape) != 1):
            raise InvalidArrayDimensionColor()
        if(not isinstance(radius, numbers.Number)):
            raise TypeError()
        if(type(illum) != Illum):
            raise TypeError()
        if(type(reflection) != float):
            raise TypeError()
        self.pos = pos
        self.illum = illum
        self.pos = pos
        self.radius = radius
        self.color = color
        self.shininess = shininess
        self.reflection = reflection

class Light:
    def __init__(self, pos : np.array, illum : Illum):
        if(len(pos) != 3):
            raise InvalidDimension()
        if(len(pos.shape) != 1):
            raise InvalidArrayDimensionPos()
        if(type(illum) != Illum):
            raise TypeError()
        self.pos = pos
        self.illum = illum

class Ray:
    normal_vector = np.array([0,1,0])

    def __init__(self, pos : np.array, vector : np.array, color : np.array):
        if(len(pos) != 3 or len(vector) != 3 or len(color) != 3):
            raise InvalidDimension()
        if(len(pos.shape) != 1):
            raise InvalidArrayDimensionPos()
        if(len(vector.shape) != 1):
            raise InvalidArrayDimensionVector()
        if(len(color.shape) != 1):
            raise InvalidArrayDimensionColor()
        self.pos = pos
        self.vector = vector/np.linalg.norm(vector)
        self.color = color
    
    def sphere_intersect(self, sphere : Sphere) -> int:
        #a = 1 #(self.vector[0]**2 + self.vector[1]**2 + self.vector[2]**2) length of line vector will be always 1
        m = self.pos[0] - sphere.pos[0]
        n = self.pos[1] - sphere.pos[1]
        l = self.pos[2] - sphere.pos[2]
        b = 2*m*self.vector[0] + 2*n*self.vector[1] + 2*l*self.vector[2]
        c = m**2 + n**2 + l**2 - sphere.radius**2
        delta = b**2 - 4*c
        if(delta > 0): #this means that we actually intersect
            t1 = (-b + np.sqrt(delta))/2
            t2 = (-b - np.sqrt(delta))/2
            if(t1 > 0 and t2 > 0):
                #always t2?
                return min(t1,t2) #when we do intersect we intersect in 2 spots, we want the smaller one because that is the one we are "seeing"
        return None
    
    def set_color_cosine_basic(self, intersection_point : np.array, object : Sphere, light_source : Light, cam_pos):
        normal_to_surface = normalize(intersection_point - object.pos)
        intersection_to_light = normalize(light_source.pos - intersection_point)
        self.color = object.color * np.clip(np.dot(intersection_to_light, normal_to_surface),0,1)

    
    def set_color_cosine_2(self, intersection_point : np.array, object : Sphere, light_source : Light, cam_pos): #doesn't use reflection
        normal = normalize(intersection_point - object.pos)
        intersection_to_light = normalize(light_source.pos - intersection_point)

        illumination = np.array([0,0,0], dtype=float)
        illumination += object.illum.ambient * light_source.illum.ambient #ambient(base illumination, regardless of position)

        #diffuse(between surface and light), independent of where we are looking from
        illumination += object.illum.diffuse * light_source.illum.diffuse * np.dot(intersection_to_light, normal)

        #specular(between eye and surface), the really shiny part
        intersection_to_cam = cam_pos - intersection_point
        cam_to_light = normalize(intersection_to_light + intersection_to_cam)
        illumination +=  object.illum.specular * light_source.illum.specular * np.dot(normal, cam_to_light) ** (object.shininess / 4) #TODO: * specular intensity of object

        self.color = object.color * np.clip(illumination, 0, 1)
    
    def bounce(self, intersection_point : np.array, object : Sphere):
        normal = normalize(intersection_point - object.pos)
        self.vector = self.vector - 2 * np.dot(self.vector, normal) * normal
        self.pos = intersection_point

    def set_color_cosine_bounce(self, intersection_point : np.array, object : Sphere, light_source : Light, cam_pos, cumulative_reflection): #uses reflection
        normal = normalize(intersection_point - object.pos)
        intersection_to_light = normalize(light_source.pos - intersection_point)

        illumination = np.array([0,0,0], dtype=float)
        illumination += object.illum.ambient * light_source.illum.ambient #ambient(base illumination, regardless of position)

        illumination += object.illum.diffuse * light_source.illum.diffuse * np.dot(intersection_to_light, normal)

        intersection_to_cam = cam_pos - intersection_point
        cam_to_light = normalize(intersection_to_light + intersection_to_cam)
        illumination +=  object.illum.specular * light_source.illum.specular * np.dot(normal, cam_to_light) ** (object.shininess / 4) #TODO: * specular intensity of object

        self.color = np.float64(self.color)
        self.color += object.color * np.clip(illumination, 0, 1) * cumulative_reflection
        return cumulative_reflection * object.reflection