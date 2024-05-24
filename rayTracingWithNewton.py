import numpy as np
import matplotlib.pyplot as plt

#Define the main color
mainColor = np.array([0.2, 0.2, 0.8])

def normalize(vector):
    return vector / np.linalg.norm(vector)


#function of a sphere
def f(v):
    return (v[0] - 0)**2 + (v[1]-0.5)**2 + (v[2]+1)**2 - 0.8**2

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

def rayrayray(f,dfdx, dfdy, dfdz, originPointT0, directionV, lightOrigin):
    
    #directionV - > the direction of the camera
    #g(t) = f(x0 + a*t, y0 + b*t, z0+ + c*t)
    #x0 is the orginalPointT0
    #directionV = [a, b, c]

    g = lambda t: f(np.array([originPointT0[0]+t*directionV[0], originPointT0[1]+t*directionV[1], originPointT0[2]+t*directionV[2]]))
    
    #and its derrivative
    gdot = lambda t: (
                      directionV[0] * dfdx(np.array([originPointT0[0]+t*directionV[0], originPointT0[1]+t*directionV[1], originPointT0[2]+t*directionV[2]])) 
                    + directionV[1] *dfdy(np.array([originPointT0[0]+t*directionV[0], originPointT0[1]+t*directionV[1], originPointT0[2]+t*directionV[2]])) 
                    + directionV[2] * dfdz(np.array([originPointT0[0]+t*directionV[0], originPointT0[1]+t*directionV[1], originPointT0[2]+t*directionV[2]]))
    )

    #number of iterations
    it=0;
    itMax=250;


    stepsize=0.1
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
    #if there has been an intersection, adjust the color by multiplying with the angle? Not sure how smart it is
    if intersection ==1 :
        print(angle)
        return angle * mainColor
    else:
        return black
            


width = 300
height = 300
pixels = np.empty(shape = (height, width, 3))


ratio = float (width) / height

cam_pos = [0,0,1]
camera = np.array(cam_pos)

lightOrigin = np.array([5,3,1])

width_pos = [-1, 1] #minimum and maximum width
height_pos = [-ratio, ratio] #minimum and maximum height

for i, y in enumerate(np.linspace(height_pos[0], height_pos[1], height)):
    for j, x in enumerate(np.linspace(width_pos[0], width_pos[1], width)):
        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)

        colorToBePlaced=rayrayray(f, dfdx, dfdy, dfdz, camera, direction, lightOrigin)
        
        pixels[i, j] = colorToBePlaced

plt.imsave('C2.png', pixels)
