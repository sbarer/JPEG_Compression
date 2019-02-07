
import argparse
from PIL import Image
import numpy as np

img = Image.open('pupper.jpg')

width, height = img.size
pixels = img.load()
print('width =  ' + str(width))


#img.show()
#find chroma block size 

y = np.zeros((height,width), dtype=np.uint8)
u = np.zeros((height/2,width/2), dtype=np.uint8)
v = np.zeros((height/2,width/2), dtype=np.uint8)
test = np.zeros((height,width,3), dtype=np.uint8)
    
for i in range(height):
    for j in range(width):
        if (i%2==0 and j%2==0):
            y[i,j] = pixels[j,i][0]
            u[i/2,j/2] = pixels[j,i][1]
            v[i/2,j/2] = pixels[j,i][2]
        else:
            y[i,j] = pixels[j,i][0]
        
        test[i,j] = pixels[j,i]


#trim padding to make array a multiple of 8
y_height, y_width = y.shape[0], y.shape[1]
u_height, u_width = u.shape[0], u.shape[1]
v_height, v_width = v.shape[0], v.shape[1]

if(u_width % 8 != 0 or u_height % 8 != 0):
    width_padding = u_width % 8
    height_padding = u_height % 8
    u = u[0:u_height - height_padding:1, 0:u_width - width_padding:1]
    v = v[0:v_height - height_padding:1, 0:v_width - width_padding:1]


    

y_block_count = (y_height * y_width) / 64
u_block_count = (u_height * u_width) / 64



print(y_block_count)
print(u_block_count)

test_img = Image.fromarray(test, 'RGB')
test_img.show()






