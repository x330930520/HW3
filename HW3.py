import numpy as np
import matplotlib.pyplot as plt

def question_1(x, y, c1, c2): # we used Python coordinates which means in a coordinate (x, y), x stands for row number and y stands for column number
    center_x = x//2
    center_y = y//2
    
    m = np.matrix([[-1,0,center_x],[0,1,center_y],[0,0,1]])#transformation matrix
    b = np.matrix([[c1],[c2],[1]])
    matrix= (m*b)
    if matrix[0] == 0:# convert 0s in coordinates since Python uses 0-based system (starts with 0 instead of 1)
        matrix[0] = 1
    if matrix[1] == 0:
        matrix[1] = 1
    #print (matrix)
    return [m, matrix]
    
def bresenham(x0, y0, x1, y1):
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0  
        x1, y1 = y1, x1

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    if y0 < y1: 
        ystep = 1
    else:
        ystep = -1

    deltax = x1 - x0
    deltay = abs(y1 - y0)
    error = -deltax / 2
    y = y0

    line = []    
    for x in range(x0, x1 + 1):
        if steep:
            line.append((y,x))
        else:
            line.append((x,y))

        error = error + deltay
        if error > 0:
            y = y + ystep
            error = error - deltax
    return line

    
def question_1_canvas():
    #create a 600*600 white canvas
    canvas = np.zeros((600,600, 3))
    canvas[:, :, 0] = np.ones((600,600))
    canvas[:, :, 1] = np.ones((600,600))
    canvas[:, :, 2] = np.ones((600,600))
    return canvas
    
def question_1_draw_axis(width, r, g, b):
    x = question_1_canvas()
    cx = x.shape[1]//2
    cy = x.shape[0]//2
    set = [[[cx, 0], [cx, x.shape[0]]], [[0, cy], [x.shape[1], cy]]]
    fig = plt.figure()
    for p in set:
        points = bresenham(p[0][0], p[0][1], p[1][0], p[1][1])
        for square in points:
            oi = square[0]
            oj = square[1]
            width_x = width
            width_y = width
            i = oi - width//2
            j = oj - width//2
            size = x.shape
            if i < 0:
                if i + width_x > x.shape[1]:
                    i = 0
                    width_x = x.shape[1]
                if i + width_x < x.shape[1] and i + width_x >= 0:
                    width_x = oi - 0.5*width + width
                    i = 0
            
            if i > 0 and i <= x.shape[1]:
                if i + width_x > x.shape[1]:
                    width_x = x.shape[1] - i
           
            if j < 0:
                if j + width_y > x.shape[0]:
                    j = 0
                    width_y = x.shape[0]
                if j + width_y < x.shape[0] and j + width_y >= 0:
                    width_y = oj - 0.5*width + width
                    j = 0
                    
            if j > 0 and j <= x.shape[0]:
                if j + width_y > x.shape[0]:
                    width_y = x.shape[0] - j
            
            
            if j > x.shape[0] or i > x.shape[1] or j + width_y <= 0 or i + width_x <= 0: #square that does not show on the canvas
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.imshow(x)
                    
            else:  #square does show on the canvas 
                x[j:j+width_y, i:i+width_x, 0] = np.ones((width_y, width_x))*r
                x[j:j+width_y, i:i+width_x, 1] = np.ones((width_y, width_x))*g
                x[j:j+width_y, i:i+width_x, 2] = np.ones((width_y, width_x))*b
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.imshow(x)
    return x

        
def question_1_square(width, r, g, b):
    x = question_1_draw_axis(2, 0, 0, 0)
    #transfer the center point to the top left point of the square
    file = []
    file_1 = np.loadtxt('xy_points.txt')
    for k in file_1:
        file.append(question_1(600, 600, k[0], k[1])[1])
    for square in file:
        oi = square[0]
        oj = square[1]
        width_x = width
        width_y = width
        i = oi - width//2
        j = oj - width//2
        size = x.shape
        if i < 0:
            if i + width_x > x.shape[1]:
                i = 0
                width_x = x.shape[1]
            if i + width_x < x.shape[1] and i + width_x >= 0:
                width_x = oi - 0.5*width + width
                i = 0
        
        if i > 0 and i <= x.shape[1]:
            if i + width_x > x.shape[1]:
                width_x = x.shape[1] - i
       
        if j < 0:
            if j + width_y > x.shape[0]:
                j = 0
                width_y = x.shape[0]
            if j + width_y < x.shape[0] and j + width_y >= 0:
                width_y = oj - 0.5*width + width
                j = 0
                
        if j > 0 and j <= x.shape[0]:
            if j + width_y > x.shape[0]:
                width_y = x.shape[0] - j
        
        
        if j > x.shape[0] or i > x.shape[1] or j + width_y <= 0 or i + width_x <= 0: #square that does not show on the canvas
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.imshow(x)
                
        else:  #square does show on the canvas 
            x[j:j+width_y, i:i+width_x, 0] = np.ones((width_y, width_x))*r
            x[j:j+width_y, i:i+width_x, 1] = np.ones((width_y, width_x))*g
            x[j:j+width_y, i:i+width_x, 2] = np.ones((width_y, width_x))*b
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.imshow(x)
    plt.show()  
    
def question_2(width, r, g, b):
    '''
    Basically code from last question
    '''
    x = question_1_draw_axis(2, 0, 0, 0)
    #transfer the center point to the top left point of the square
    file = []
    m = np.matrix([[3/5,0,0],[0,5/4,0],[0,0,1]])
    file_1 = np.loadtxt('xy_points.txt')
    for k in file_1:
        file.append(m*question_1(600, 600, k[0], k[1])[1])
    for square in file:
        oi = square[0]
        oj = square[1]
        width_x = width
        width_y = width
        i = oi - width//2
        j = oj - width//2
        size = x.shape
        if i < 0:
            if i + width_x > x.shape[1]:
                i = 0
                width_x = x.shape[1]
            if i + width_x < x.shape[1] and i + width_x >= 0:
                width_x = oi - 0.5*width + width
                i = 0
        
        if i > 0 and i <= x.shape[1]:
            if i + width_x > x.shape[1]:
                width_x = x.shape[1] - i
       
        if j < 0:
            if j + width_y > x.shape[0]:
                j = 0
                width_y = x.shape[0]
            if j + width_y < x.shape[0] and j + width_y >= 0:
                width_y = oj - 0.5*width + width
                j = 0
                
        if j > 0 and j <= x.shape[0]:
            if j + width_y > x.shape[0]:
                width_y = x.shape[0] - j
        
        
        if j > x.shape[0] or i > x.shape[1] or j + width_y <= 0 or i + width_x <= 0: #square that does not show on the canvas
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.imshow(x)
                
        else:  #square does show on the canvas 
            x[j:j+width_y, i:i+width_x, 0] = np.ones((width_y, width_x))*r
            x[j:j+width_y, i:i+width_x, 1] = np.ones((width_y, width_x))*g
            x[j:j+width_y, i:i+width_x, 2] = np.ones((width_y, width_x))*b
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.imshow(x)
    plt.show()  
    


def question_3(p, width, r, g, b):
    x = question_1_canvas()
    #transfer the center point to the top left point of the square
    for t in range(len(p)-1):
            points = bresenham(p[t][0], p[t][1], p[t+1][0], p[t+1][1])
            for square in points:
                oi = square[0]
                oj = square[1]
                #width = square[2]
                width_x = width
                width_y = width
                i = oi - 0.5*width
                j = oj - 0.5*width
                size = x.shape
                '''
                Dealing with multiple situation
                '''
                if i < 0:
                    if i + width_x > x.shape[1]:
                        i = 0
                        width_x = x.shape[1]
                    if i + width_x < x.shape[1] and i + width_x >= 0:
                        width_x = oi - 0.5*width + width
                        i = 0
                
                if i > 0 and i <= x.shape[1]:
                    if i + width_x > x.shape[1]:
                        width_x = x.shape[1] - i
               
                if j < 0:
                    if j + width_y > x.shape[0]:
                        j = 0
                        width_y = x.shape[0]
                    if j + width_y < x.shape[0] and j + width_y >= 0:
                        width_y = oj - 0.5*width + width
                        j = 0
                        
                if j > 0 and j <= x.shape[0]:
                    if j + width_y > x.shape[0]:
                        width_y = x.shape[0] - j
                
                
                if j > x.shape[0] or i > x.shape[1] or j + width_y <= 0 or i + width_x <= 0: #square that does not show on the canvas
                    plt.xlabel("X")
                    plt.ylabel("Y")
                    plt.imshow(x)
                        
                else:  #square does show on the canvas 
                    x[j:j+width_y, i:i+width_x, 0] = np.ones((width_y, width_x))*r
                    x[j:j+width_y, i:i+width_x, 1] = np.ones((width_y, width_x))*g
                    x[j:j+width_y, i:i+width_x, 2] = np.ones((width_y, width_x))*b
                    plt.xlabel("X")
                    plt.ylabel("Y")
                    plt.imshow(x)
    #plt.show() 
    name = "question_3_a2.jpg"
    plt.savefig(name) 



def question_3_sheer(a):
    points = [[0,0], [0,200], [200,200], [250,100], [200,0], [0,0]]
    sheer = np.matrix([[1, a], [0, 1]])
    lst = []
    for x in points:
        lst.append((question_1(600, 600, x[0], x[1])[1][:2].reshape(1,2)*sheer).tolist())
    new = []
    for x in lst:
        new.append([x[0][1], x[0][0]])
    question_3(new, 10, 0, 0, 1)
    
    

#question_1(401, 500, 0, 0)
#question_1(300, 451, 200, 300)
#question_1(600, 600, -300, -200)

#question_1_square(21, 0, 0, 1)
#question_2(21, 0, 1, 0)


'''
points = [[0,0], [0,200], [200,200], [250,100], [200,0], [0,0]]
lst = []
for x in points:
        #temp = []
        lst.append(question_1(600, 600, x[0], x[1])[1][:2].tolist())
        #lst.append(temp)
new = []
for x in lst:
    new.append([x[1][0], x[0][0]])
question_3(new, 10, 0, 0, 1)
'''
#question_3_sheer(1)

#question_3_sheer(2)

    
