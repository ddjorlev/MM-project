import numpy as np
import matplotlib.pyplot as plt
import time
#Define the main color
#mainColor = np.array([0.2, 0.2, 0.8])

def normalize(vector):
    return vector / np.linalg.norm(vector)

light_ambient = np.array([0.6,0.6,0.6])
light_specular = np.array([1,1,1])
light_diffuse = np.array([1,1,1])

#function of a sphere
def f1(v):
    return (v[0] + 0.1)**2 + (v[1]-0.5)**2 + (v[2]+1)**2 - 0.7**2

def f2(v):
    return (v[0] - 0.2)**2 + (v[1]+0.3)**2 + (v[2]+2)**2 - 0.7**2

def f3(v):
    return (v[0] - 0)**2 + (v[1]+1000)**2 + (v[2]+0)**2 - 999**2

def get_illum(func):
    #ambient, diffuse, specular, shininess, reflection
    if func == f1:
        return np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1]) , 80, 0.4
    if func == f2:
        return np.array([1,1,1]), np.array([1,1,1]), np.array([1,1,1]) , 80, 0.4
    if func == f3:
        return np.array([0.7,0.7,0.7]), np.array([0.6,0.6,0.6]), np.array([0.1,0.1,0.1]) , 80, 0.1
    
def get_color(func):
    if func == f1:
        return np.array([57/256, 68/256, 188/256])
    if func == f2:
        return np.array([123/256, 6/256, 35/256])
    if func == f3:
        return np.array([150/256, 70/256, 150/256])
    
def get_center(func):
    if func == f1:
        return np.array([-0.10, +0.5, -1])
    if func == f2:
        return np.array([0.2, -0.3, -2])
    if func == f3:
        return np.array([0, -1000, 0])


#derrivatives 
def dfdx(v):
    return 2 * (v[0])
def dfdy(v): 
    return 2 * (v[1]- 0.5) 
def dfdz(v): 
    return 2 * (v[2] + 1)

#Newton method
def newton(f,Df,x0,epsilon,max_iter):
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)

        if abs(fxn) < epsilon:
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0: 
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    
    return xn

def sign(f, point):
    s = np.sign(f(point))
    if s == 0:
        s=1
    return s

#reflection which returns the cos of the angle between the norm and the light source
def reflection(v, T, lightOrigin, dfdx, dfdy, dfdz):
   
    n=np.array([dfdx(T), dfdy(T), dfdz(T)])
    n=normalize(n)

    Firstvec= lightOrigin-T

    result = np.dot(n,Firstvec) / (np.linalg.norm(n) * np.linalg.norm(Firstvec)) 
    result = np.cos(result)
    if result <0:
        print("the is a negative angle")
    return result

def rayrayray(f,dfdx, dfdy, dfdz, originPointT0, directionV, lightOrigin, cumulative_reflection):

    g = lambda t: f(np.array([originPointT0[0]+t*directionV[0], originPointT0[1]+t*directionV[1], originPointT0[2]+t*directionV[2]]))
    
    #and its derrivative
    gdot = lambda t: (
                      directionV[0] * dfdx(np.array([originPointT0[0]+t*directionV[0], originPointT0[1]+t*directionV[1], originPointT0[2]+t*directionV[2]])) 
                    + directionV[1] *dfdy(np.array([originPointT0[0]+t*directionV[0], originPointT0[1]+t*directionV[1], originPointT0[2]+t*directionV[2]])) 
                    + directionV[2] * dfdz(np.array([originPointT0[0]+t*directionV[0], originPointT0[1]+t*directionV[1], originPointT0[2]+t*directionV[2]]))
    )

    #number of iterations
    it=0
    itMax=250


    stepsize=0.01
    t=stepsize 
    
    stepDirection = stepsize * directionV
    
    prevPoint = originPointT0
    
    #calculating the next point
    newPoint = prevPoint + stepsize
    
    #previous sing and the current sign
    prevSign = sign(f, originPointT0)
    currSign = sign(f, newPoint) 

    #to check if there has been an intersection
    intersection=0
    angle=0
    
    while(it<itMax):
        
        it=it+1
        t=t+stepsize
        
        prevPoint= newPoint
        newPoint = newPoint + stepDirection

        prevSign=currSign
        currSign=sign(f, newPoint)

        u = 0
        betterPoint = newPoint
        if currSign != prevSign:
            
            #we start at the middle of the interval t=(t1+t2)/2 following the instructions in the pdf
            startOfApproximation = ( t + (t-stepsize)) /2

            #returns the exact point
            u = newton(g, gdot, startOfApproximation, 1e-10, 100)
            betterPoint = originPointT0 + u*directionV
            
            #Here should send another ray

            intersection=1
            
            angle=reflection(directionV, betterPoint, lightOrigin, dfdx, dfdy, dfdz)
            
            break
            
    black = np.array([0.0, 0.0, 0.0])
    colorToAdd = np.array([0.0, 0.0, 0.0])
    ambient, diffuse, specular, shininess, obj_reflection = get_illum(f)
    colorToAdd += ambient * light_ambient

    normal = normalize(betterPoint - get_center(f))
    intersection_to_light = normalize(lightOrigin - betterPoint)
    colorToAdd += diffuse * light_diffuse * np.dot(intersection_to_light,normal)
    intersection_to_cam = camera - betterPoint
    cam_to_light = normalize(betterPoint + intersection_to_cam)
    colorToAdd += specular * light_specular * np.dot(normal, cam_to_light) ** (shininess/4)
    #if there has been an intersection, adjust the color by multiplying with the angle? Not sure how smart it is
    if intersection ==1 :
        #print(angle)
        #return angle * mainColor, betterPoint
        return get_color(f) * np.clip(colorToAdd, 0, 1) * cumulative_reflection, betterPoint, obj_reflection
    else:
        return black, np.array([np.inf, np.inf, np.inf]), 1

width = 300
height = 300
pixels = np.empty(shape = (height, width, 3))


ratio = float (width) / height

cam_pos = [0,0,1]
camera = np.array(cam_pos)

lightOrigin = np.array([5,3,1])

width_pos = [-1, 1] #minimum and maximum width
height_pos = [-ratio, ratio] #minimum and maximum height

functions = [f1, f2, f3]
start_time = time.time()
# samples = 2
# for k in range(samples):
for i, y in enumerate(np.linspace(height_pos[0], height_pos[1], height)):
    for j, x in enumerate(np.linspace(width_pos[0], width_pos[1], width)):
        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)
        cumulative_reflection = 1
        for bounce in range(5):
            t_min = np.inf
            f_min = 0
            point_min = 0
            for func in functions:
                colorToBePlaced, betterPoint, _ = rayrayray(func, dfdx, dfdy, dfdz, camera, direction, lightOrigin, cumulative_reflection)
                temp = np.linalg.norm(betterPoint - origin)
                if temp < t_min and temp != np.inf:
                    t_min = temp 
                    f_min = func
                    point_min = betterPoint
                # if(f_min == f3):
                #     print(f_min)
                # print(point_min)
            #print(t_min)
            if f_min != 0:
                colorToBePlaced, betterPoint, obj_reflection = rayrayray(f_min, dfdx, dfdy, dfdz, camera, direction, lightOrigin, cumulative_reflection)
                cumulative_reflection *= obj_reflection
            else:
                colorToBePlaced = np.array([0,0,0])
            # colorToBePlaced, t = rayrayray(f1, dfdx, dfdy, dfdz, camera, direction, lightOrigin)
            if(f_min != 0):
                normal = normalize(betterPoint - get_center(f_min))
                direction = direction - 2 * np.dot(direction, normal) * normal
                origin = betterPoint
            pixels[height-i-1, j-1] += colorToBePlaced
            if(f_min == 0):
                break

# for i, y in enumerate(np.linspace(height_pos[0], height_pos[1], height)):
#     for j, x in enumerate(np.linspace(width_pos[0], width_pos[1], width)):
#         pixels[width-i-1, j-1] = pixels[width-i-1, j-1]/samples

print("time taken: ", time.time() - start_time)
plt.imsave('C200.png', pixels)
print("time taken: ", time.time() - start_time)