
import argparse
from PIL import Image
import numpy as np
import math
#To grab arguments from input

def file_inputs():
    import argparse
    #Calls an instance of the argument Parser
    parser = argparse.ArgumentParser(description = 'Process input file path and export file path')
    #add arguments for input file and output file
    #Positional Arguments
    parser.add_argument('input_file', type=str, help='relative path to the input image from current directory')
    parser.add_argument('output_file', type=str, help='relative path to the input image from current directory')
    #parse through arguments and put into arg
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    return input_file, output_file

#Takes an image in file_path and retuns image with rgb->yuv pixel value conversion
def rgb_to_yuv(file_path):
    img = Image.open(file_path)
    pixels = img.load() #Create Pixel map

    width = img.size[0]
    height = img.size[1]

    #RESIZE IMAGE IF IT IS not a scalar multiple of 8x8

    if (width%8 != 0 or height%8 !=0):
        #Splice original image
        width_padding = width%8
        height_padding = height%8
        allpix = np.array(img, dtype=np.uint8)
        temp = allpix[0:width-width_padding:1, 0:height-height_padding:1]
        #load spliced image into pixeldata
        image = Image.fromarray(temp, 'RGB')
        pixels = image.load()
        width = image.size[0]
        height = image.size[1]


    #RGB -> YUV conversion
    for i in range(width):
        for j in range(height):
            r = pixels[i, j][0]
            g = pixels[i, j][1]
            b = pixels[i, j][2]
            y = 0.299*r + 0.587*g + 0.114*b
            u = 0.492* (b-y)
            v = 0.877 * (r-y)
            pixels[i,j] = (int(y),int(u),int(v))
    return img

#Recover UV values using single pass
#WORK ON THIS
def recover_uv(npmat):


    return 'To be implemented soon!' + npmat




#applys a 4:2:0 chroma subsampling to image
#returns 3 2D arrays corresponding to YUV
#FORMAT: pixels[width, height] np.array[height, width]
def chroma_ss_process(img):
    width = img.size[0]
    height = img.size[1]
    pixels = img.load()
    #find chroma block size
    y = np.zeros((height, width), dtype=np.uint8)
    u = np.zeros((height/2, width/2), dtype=np.uint8)
    v = np.zeros((height/2, width/2), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if (i%2 == 0 and j%2==0):
                y[i, j] = pixels[j, i][0]
                u[i/2, j/2] = pixels[j, i][1]
                v[i/2, j/2] = pixels[j, i][2]
            else:
                y[i, j] = pixels[j, i][0]

    #trim padding to make array a multiple of 8
    u_height, u_width = u.shape[0], u.shape[1]
    v_height, v_width = v.shape[0], v.shape[1]

    if(u_width % 8 != 0 or u_height % 8 != 0):
        width_padding = u_width % 8
        height_padding = u_height % 8
        u = u[0:u_height - height_padding:1, 0:u_width - width_padding:1]
        v = v[0:v_height - height_padding:1, 0:v_width - width_padding:1]

    return y, u, v

def initialize_DCT_matrix():
    a = 1 / (2 * math.sqrt(2))
    matrix = np.empty([8,8])

    for i in range(0, 8, 1):
        for j in range(0, 8, 1):
            if (i == 0):
                matrix[i, j] = a
            else:
                matrix[i, j] = 0.5 * math.cos((2*j + 1)*i*math.pi / 16)

    return matrix

def DCT(comp, r, c, T):
    # initialize T transpose from T
    TT = np.transpose(T)
    # create new array for DCT'd values
    Fcomp = np.empty([r,c])

    # iterate through comp in 8x8 blocks
    for i in range(0, r-8, 8):
        for j in range(0, c-8, 8):
            block = comp[i:i+8, j:j+8]

            # apply DCT algorithm to 8x8 block
            Fblock = T * block * TT

            # add block to Fcomp matrix
            Fcomp[i:i + 8, j:j + 8] = Fblock

    return Fcomp

def initialize_Q_LUM():
    lum = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 36, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]])

    return lum

def initialize_Q_CHROME():
    chrome = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                       [18, 21, 26, 66, 99, 99, 99, 99],
                       [24, 26, 56, 99, 99, 99, 99, 99],
                       [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])

    return chrome


def Quantize(comp, r, c, Q):
    FQ = np.empty([r,c])
    for i in range(0, r, 1):
        for j in range(0, c, 1):
            FQ[i, j] = comp[i, j] / Q[i, j]

    FQ = round(comp / Q)

    return FQ


def main():
    input , output = file_inputs()
    print("input file name = " + str(input))
    print("output file name = " + str(output))

    # 1 - convert RGB to YUV, resize to fit 8x8
    image = rgb_to_yuv(input)

    # 2 - Split YUV into Y U V and subsequent downsampling
    y, u, v = chroma_ss_process(image)

    #Create a 2D matrix from subsampling. Numpy Matrix multiplication will be used here because
    #it is more efficient in terms of computation and space. Compression will be implemented on array
    #Then the final decompressed array will be the vector values for the image to be rendered.

    # 3 - Calculate dimensions of Y U V
    rows, cols = y.shape[0], y.shape[1]
    uv_rows, uv_cols = u.shape[0], u.shape[1]
    print("rows = " + str(rows))
    print("cols = " + str(cols))

    # Confirm all arrays are divisible by 8
    y_blockcount = (rows * cols) / 64
    uv_blockcount = (uv_rows * uv_cols) / 64

    # 4 - DCT
    DCT_matrix = initialize_DCT_matrix()

    Fy = DCT(y, rows, cols, DCT_matrix)
    Fu = DCT(u, uv_rows, uv_cols, DCT_matrix)
    Fv = DCT(v, uv_rows, uv_cols, DCT_matrix)

    # 5 - Quantization
    Q_matrix_LUM = initialize_Q_LUM()
    Q_matrix_CHROME = initialize_Q_CHROME()

    FQy = Quantize(Fy, rows, cols, Q_matrix_LUM)
    FQu = Quantize(Fu, uv_rows, uv_cols, Q_matrix_CHROME)
    FQv = Quantize(Fv, uv_rows, uv_cols, Q_matrix_CHROME)







    #Iterate through array in an 8x8 block
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            for k in range(3):
                if(len(y[i, j] == 1)):
                    pass
                else:
                    #scale data to center around 0
                    #Make sure to add padding to image i.e width/height % 8 != 0
                    #block = npmat[i:i+8, j:j+8, k] - 128
                    pass


    ###CHROMOSUBSAMPLING
    ## How do to the 4:2:0 subsampling

    #img = Image.fromarray(block, 'RGB')
    #print(block)
    #img.show()
    #ss_image.show()







#Checks to see if this is the main module being run
if __name__ == '__main__':
    main()


