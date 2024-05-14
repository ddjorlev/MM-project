import numpy as np
from classes import Ray, Sphere, InvalidDimension, InvalidArrayDimensionVector, InvalidArrayDimensionPos 
import unittest

class Tester(unittest.TestCase):
    def testRaises(self, exception : Exception, func):
        self.assertRaises(exception, func)
    def testNoRaise(self, func):
        try:
            func()
        except:
            return self.fail() 
    def testAlmostEqual(self, first, second):
        self.assertAlmostEqual(first, second, places = 8)

def initalization_test_1():
    ray = Ray(np.array([]), np.array([]))
    sphere = Sphere(np.array([]), np.array([]))

def initalization_test_2():
    ray = Ray(np.array([1,2,3]), np.array([100,299,334]))
    sphere = Sphere(np.array([[1],[2],[3]]), np.array([5,6,7]))

def initalization_test_3():
    ray = Ray(np.array([5,6,7]), np.array([1,1,1]))
    sphere = Sphere(np.array([1,2,3]), np.array([6,7,8]))

def initalization_test_4():
    ray = Ray(np.array([5,6,7]), np.array([1,1,1]))
    sphere = Sphere(np.array([17141412,29509,1231251]), 420)

def intersection_test_1():
    ray = Ray(np.array([0,4,0]), np.array([0,-1,0]))
    sphere = Sphere(np.array([0,0,0]), 2)
    return ray.sphere_intersect(sphere) #t=2

def intersection_test_2():
    ray = Ray(np.array([0,10,0]), np.array([1,1,1]))
    sphere = Sphere(np.array([1,2,3]), 2)
    return ray.sphere_intersect(sphere) #doesn't intersect

def intersection_test_3():
    ray = Ray(np.array([13,-6,0]), np.array([1,-1,0]))
    sphere = Sphere(np.array([3,2,0]), 2)
    return ray.sphere_intersect(sphere) #sphere behind camera

def intersection_test_4():
    ray = Ray(np.array([13,-6,0]), np.array([-1/np.sqrt(2),1/np.sqrt(2),0]))
    sphere = Sphere(np.array([3,2,0]), 2)
    return ray.sphere_intersect(sphere) #t=8 root 2

def intersection_test_5():
    ray = Ray(np.array([1,1,1]), np.array([4,3,2]))
    sphere = Sphere(np.array([6,5,4]), np.sqrt(3))
    return ray.sphere_intersect(sphere) 

tester = Tester()

tester.testRaises(InvalidDimension, initalization_test_1)
tester.testRaises(InvalidArrayDimensionPos, initalization_test_2)
tester.testRaises(TypeError, initalization_test_3)
tester.testNoRaise(initalization_test_4)

tester.testAlmostEqual(2, intersection_test_1())
tester.testAlmostEqual(None, intersection_test_2())
tester.testAlmostEqual(None, intersection_test_3()) 
tester.testAlmostEqual(8*np.sqrt(2), intersection_test_4()) 
tester.testAlmostEqual(5.3851648071, intersection_test_5()) #calculated manually with wolfram alpha

