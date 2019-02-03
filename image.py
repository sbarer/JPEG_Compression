from PIL import Image

img = Image.open('lena.ppm')
pixels = img.load() #Create Pixel map

width = img.size[0]
height = img.size[1]

#RGB -> YUV conversion 
for i in range(width):
    for j in range(height):
        r = pixels[i,j][0]
        g = pixels[i,j][1]
        b = pixels[i,j][2]
        y = 0.299*r + 0.587*g + 0.114*b
        u = 0.492* (b-y)
        v = 0.877 * (r-y)
        pixels[i,j] = (int(y),int(u),int(v))

for i in range(width):
    print(i)