
from PIL import Image
import numpy.matlib
import numpy as np
import math
# To grab arguments from input

def file_inputs():

    import argparse
    # Calls an instance of the argument Parser
    parser = argparse.ArgumentParser(description = 'Process input file path and export file path')
    # add arguments for input file and output file
    # Positional Arguments
    parser.add_argument('input_file', type=str, help='relative path to the input image from current directory')
    parser.add_argument('output_file', type=str, help='relative path to the input image from current directory')
    # parse through arguments and put into arg
    args = parser.parse_args()

    input_file_path = args.input_file
    output_file_path = args.output_file


    return input_file_path, output_file_path

# Takes an image in file_path and retuns image with rgb->yuv pixel value conversion
def rgb_to_yuv(file_path):
    img = Image.open(file_path)
    pixels = img.load()  # Create Pixel map

    width = img.size[0]
    height = img.size[1]

    # RESIZE IMAGE IF IT IS not a scalar multiple of 8x8

    if (width % 8 != 0 or height % 8 !=0):
        # Splice original image
        width_padding = width%8
        height_padding = height%8
        allpix = np.array(img, dtype=np.uint8)
        temp = allpix[0:width-width_padding:1, 0:height-height_padding:1]
        # load spliced image into pixeldata
        image = Image.fromarray(temp, 'RGB')
        pixels = image.load()
        width = image.size[0]
        height = image.size[1]

    # RGB -> YUV conversion
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

# Recouver UV values using single pass
# WORK ON THIS
#array[rows, cols]
# rows, cols = array.shape
def recouver_uv(small, lrows,lcols):
    rows, cols = small.shape
    large = np.zeros((lrows,lcols), dtype=np.uint8)
    lrows, lcols = large.shape

    col_padding = abs(lcols - (cols * 2))
    row_padding = abs(lrows - (rows * 2))
    #print('col_padding: ',col_padding)
    #print('row_padding: ' ,row_padding)
    r = lrows -row_padding
    c = lcols -col_padding
    #Initial fill of large matrix
    for i in range(0,r):
        for j in range(0,c):
            #print(i,j)
            if (i % 2 ==0 and j %2 ==0):    #top left corner
                #print('i,j:',i,j)
                #print('padding values: ',row_padding,col_padding)
                #print('dimensions: ',r,c)
                large[i,j] = small[i/2,j/2]
            
        
    #first interpolation pass
    last_col = lcols-1
    #print('this is last col', last_col)
    for i in range(0,lrows,2):
        for j in range(0,lcols):
            #print(i,j)
            if (i % 2 ==0 and j %2 ==0): 
                pass
            else:    
                if(j==int((last_col))):   #if the current element is at the very last colum
                    large[i,j] = large[i,j-1]
                else:
                    #print(i,j)
                    avg_val = (large[i,j-1] + large[i, j+1])/2  #avg val of two columns
                    avg_val = round(avg_val)
                    large[i,j] = avg_val

    #print(large)
                
    #second interpolation pass
    last_row = lrows - 1
    for i in range(1,lrows,2):
        for j in range(0,lcols):
            if(i== last_row):
                large[i,j] = large[i-1,j]
            else:
                avg_val = (large[i-1,j] + large[i+1,j])/2
                avg_val = round(avg_val)
                large[i,j] = avg_val

    #input small array values into large array 



    #print('small', small)
    #print('large', large)
    #print(rows, cols)
    return large
    


def yuv_merge(y, u, v, rows, cols):
    merged_yuv = np.zeros((rows, cols,3), dtype=np.uint8)
    for i in range(0,rows):
        for j in range(0,cols):
            merged_yuv[i,j][0] = int(y[i,j])
            merged_yuv[i,j][1] = int(u[i,j])
            merged_yuv[i,j][2] = int(v[i,j])

    return merged_yuv


def yuv_to_rgb(matrix):
    rgb_converter = np.array([[1.0, 1.0, 1.0],
                  [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                  [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])

    rgb_matrix = np.dot(matrix, rgb_converter)
    rgb_matrix[:, :, 0] -= 179.45477266423404
    rgb_matrix[:, :, 1] += 135.45870971679688
    rgb_matrix[:, :, 2] -= 226.8183044444304

    return rgb_matrix
#TODO: write the converter
def yuv_to_rgb2(matrix):
    row, col = matrix.shape
    rgb_matrix = np.zeros((row,col), dtype=np.uint8)
    for i in range(row):
        for j in range(col): 
            pass
            #TODO Write the conver
            
    return rgb_matrix

# applys a 4:2:0 chroma subsampling to image
# returns 3 2D arrays corresponding to YUV
# FORMAT: pixels[width, height] np.array[height, width]
def chroma_ss_process(img):
    width = img.size[0]
    height = img.size[1]
    pixels = img.load()
    # find chroma block size
    y = np.zeros((height, width), dtype=np.uint8)
    h = height // 2
    w = width // 2
    u = np.zeros((h, w), dtype=np.uint8)
    v = np.zeros((h, w), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if (i % 2 == 0 and j % 2==0):
                y[i, j] = pixels[j, i][0]
                u[i//2, j//2] = pixels[j, i][1]
                v[i//2, j//2] = pixels[j, i][2]
            else:
                y[i, j] = pixels[j, i][0]

    # trim padding to make array a multiple of 8
    u_height, u_width = u.shape[0], u.shape[1]
    v_height, v_width = v.shape[0], v.shape[1]

    if(u_width % 8 != 0 or u_height % 8 != 0):
        width_padding = u_width % 8
        height_padding = u_height % 8
        u = u[0:u_height - height_padding:1, 0:u_width - width_padding:1]
        v = v[0:v_height - height_padding:1, 0:v_width - width_padding:1]

    #print('u', u)
    #print('dimensions', u.shape)
    return y, u, v


def initialize_DCT_matrix():
    a = round(float(1 / (2 * math.sqrt(2))),5)
    matrix = np.empty([8, 8])

    for i in range(0, 8, 1):
        for j in range(0, 8, 1):
            if (i == 0):
                matrix[i, j] = a
            else:
                matrix[i, j] = round(float(0.5 * np.cos((((2*j) + 1) * i * numpy.pi) / 16)),5)

    print("DCT Matrix:")
    print(matrix)

    return matrix

#Perfroms DCT to pixels of an image
#Params: r - is the number of rows in pixels
#        c - is the number of colums in pixels
#        DCT - matrix 
#        Q - is Quantizaiton matrix 
def apply_DCT(pixels, r, c, DCT_matrix, Q):
    # initialize T transpose from T
    DCT_matrix_transposed = np.transpose(DCT_matrix)

    # create new array for DCT'd values
    #Fcomp is the final array with DCT applied to it
    #FQcomp the Fcomp with quantization applied 
    Fcomp = np.empty([r, c])
    FQcomp = np.empty([r, c])

    r_padding = r%8
    c_padding = c%8
    r = r - r_padding
    c = c- c_padding
    print('this is r outside of the loop', r)
    print('this is c outside of the loop', c)
    # iterate through comp in 8x8 blocks
    for i in range(0, r, 8):
        for j in range(0, c, 8):
            #print('i,j', i,j)
            #Create block of image pixels
            block = np.empty([8,8])
            block_row, block_col = block.shape

            #Create Block of 8x8 pixel values to perform DCT on
            for x in range(0,8,1):
                for y in range(0,8,1):
                    block[y,x] = pixels[i+x,j+y]
            

            # apply DCT algorithm to 8x8 block
            #custom Matrix multiplication
            #Row summation 
            Fblock = np.zeros((8, 8),  dtype=np.float64)
            FFblock = np.zeros((8, 8), dtype=np.float64)
            for p in range(0, 8, 1):
                for a in range(0, 8, 1):
                    for b in range(0, 8, 1):
                        # multiply T and block
                        
                        Fblock[p, b] += (DCT_matrix[p, a] * (block[a, b]-128))
            #Column summation 
            for p in range(0, 8, 1):
                for a in range(0, 8, 1):
                    for b in range(0, 8, 1):
                        # multiply T and block
                        FFblock[p, b] += (Fblock[p, a] * DCT_matrix_transposed[a, b])

            # add block to Fcomp matrix
            for x in range (i, i+8, 1):
                for y in range(j, j + 8, 1):
                    Fcomp[x][y] = FFblock[(x % 8)][(y % 8)]

            # Apply Quantization to block
            Qblock = np.empty([8, 8])
            for k in range(0, 8, 1):
                for L in range(0, 8, 1):
                    Qblock[k, L] = round(FFblock[k, L] / Q[k, L])

            # add block to FQcomp matrix
            for x in range (i, i+8, 1):
                for y in range(j, j + 8, 1):
                    FQcomp[x][y] = Qblock[(x%8)][(y%8)]
            #FQcomp[i:i + 8][j:j + 8] = Qblock
            #print(Qblock)

    #TODO: take care of padding issues when Y is not a clean 8x8 block 
    #print(FQcomp[0,589])
    #print(FQcomp[589,0])

    return FQcomp



def inverse_DCT(pixels, rows, cols, DCT_matrix, Q):
    # initialize T transpose from T
    DCT_matrix_transposed = np.transpose(DCT_matrix)

    # create new array for iDCT'd values
    iFcomp = np.empty([rows, cols])
    iFQcomp = np.empty([rows, cols])

    rows_padding = rows % 8 
    cols_padding = cols % 8
    r = rows - rows_padding
    c = cols - cols_padding
    #print('r,c',r,c)
    # iterate through pixels that have undergone compression in 8x8 blocks
    for i in range(0, r, 8):
        for j in range(0, c, 8):
            
            iblock = np.empty([8,8])
            for x in range(0,8,1):
                for y in range(0,8,1):
                    #print('pixels access numbers', i+x, j+y)
                    #print('pixel access test', pixels[i+x, j+y])
                    iblock[x,y] = pixels[i+x, j+y]
                    
            #print('iblock test', iblock)
            # Apply reverse Quantization to 8x8 block of compressed pixels
            iQblock = np.empty([8, 8])
            for k in range(0, 8, 1):
                for L in range(0, 8, 1):
    
                    iQblock[k, L] = round(iblock[k, L] * Q[k, L])
            #print('iQblock after reverse quantization', iQblock )        
            #print('exited reverse quantization')
            
            #insert 8x8 pixels into to be returned np array that has values decompressed
            for x in range(0,8,1):
                for y in range(0,8,1):
                    iFcomp[i+x,j+y] = iQblock[x,y]

            # apply iDCT algorithm to 8x8 block
            iFblock = np.zeros((8, 8),  dtype=np.float64)
            iFFblock = np.zeros((8, 8), dtype=np.float64)
            for p in range(0, 8, 1):
                for a in range(0, 8, 1):
                    for b in range(0, 8, 1):
                        # multiply T and block
                        iFblock[p, b] += (DCT_matrix_transposed[p, a] * (iQblock[a, b]))

            for p in range(0, 8, 1):
                for a in range(0, 8, 1):
                    for b in range(0, 8, 1):
                        # multiply T and block
                        iFFblock[p, b] += (iFblock[p, a] * DCT_matrix[a, b])

            #apply scaling to block
            #for x in range(0,8,1):
                #for y in range(0,8,1):
                    #iFcomp[i+x,j+y] = iFFblock[x][y]
            #iFcomp[i:i + 8][j:j + 8] = iFFblock

            for x in range(0,8,1):
                for y in range(0,8,1):
                    iFFblock[x,y] = round(iFFblock[x,y] + 128 )

            for x in range(0, 8, 1):
                for y in range(0, 8, 1):
                    iFcomp[i+x , j+y] = iFFblock[x,y]

            #print('iFcomp', iFcomp[i:i+8,j:j+8])
            
    
    return iFcomp


def initialize_Q_LUM(Q_fact):
    lum = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 36, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]])

    for a in range(0, 8, 1):
        for b in range(0, 8, 1):
            lum[a, b] = lum[a, b] * Q_fact


    return lum


def initialize_Q_CHROME(Q_fact):
    chrome = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                       [18, 21, 26, 66, 99, 99, 99, 99],
                       [24, 26, 56, 99, 99, 99, 99, 99],
                       [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])

    for a in range(0, 8, 1):
        for b in range(0, 8, 1):
            chrome[a, b] = chrome[a, b] * Q_fact

    return chrome


def error(original, compressed, r, c):
    Error = np.empty([r, c])

    for a in range(0, r, 1):
        for b in range(0, c, 1):
            Error[a, b] = (original[a, b] - compressed[a, b])

    return Error


def main():
    '''
    np.set_printoptions(threshold=np.inf)
    input , output = file_inputs()
    print("input file name = " + str(input))
    print("output file name = " + str(output))
    '''

    # 1 - convert RGB to YUV, resize to fit 8x8
    input = 'pupper.jpg'
    image = rgb_to_yuv(input)

    # 2 - Split YUV into Y U V and subsequent downsampling
    y, u, v = chroma_ss_process(image)

    # Create a 2D matrix from subsampling. Numpy Matrix multiplication will be used here because
    # it is more efficient in terms of computation and space. Compression will be implemented on array
    # Then the final decompressed array will be the vector values for the image to be rendered.

    # 3 - Calculate dimensions of Y U V and Confirm all arrays are divisible by 8
    rows, cols = y.shape[0], y.shape[1]
    uv_rows, uv_cols = u.shape[0], u.shape[1]
    print("rows = " + str(rows))
    print("cols = " + str(cols))

    y_blockcount = (rows * cols) / 64
    uv_blockcount = (uv_rows * uv_cols) / 64

    test = np.array([[200, 202, 189, 188, 189, 175, 175, 175],
            [200, 203, 198, 188, 189, 182, 178, 175],
            [203, 200, 200, 195, 200, 187, 185, 175],
            [200, 200, 200, 200, 197, 187, 187, 187],
            [200, 205, 200, 200, 195, 188, 187, 175],
            [200, 200, 200, 200, 200, 190, 187, 175],
            [205, 200, 199, 200, 191, 187, 187, 175],
            [210, 200, 200, 200, 188, 185, 187, 186]])

    # 4 - DCT and Quantization
    DCT_matrix = initialize_DCT_matrix()
    quant_num = 0.5
    Q_matrix_LUM = initialize_Q_LUM(quant_num)
    Q_matrix_CHROME = initialize_Q_CHROME(quant_num)

    compressed_y_pixels = apply_DCT(y, rows, cols, DCT_matrix, Q_matrix_LUM)
    compressed_u_pixels = apply_DCT(u, uv_rows, uv_cols, DCT_matrix, Q_matrix_CHROME)
    compressed_v_pixels = apply_DCT(v, uv_rows, uv_cols, DCT_matrix, Q_matrix_CHROME)

    print('compressed y block: ',compressed_y_pixels[0:8,0:8])
    # 5 - Decode Pixel values 
    decompressed_y_pixels = inverse_DCT(compressed_y_pixels, rows, cols, DCT_matrix, Q_matrix_LUM)
    decompressed_u_pixels = inverse_DCT(compressed_u_pixels, uv_rows, uv_cols, DCT_matrix, Q_matrix_CHROME)
    decompressed_v_pixels = inverse_DCT(compressed_v_pixels, uv_rows, uv_cols, DCT_matrix, Q_matrix_CHROME)
    print('decompressed y block: ',decompressed_y_pixels[0:8,0:8])
        
    # 6 - Recover UV pixels lost from chroma subsampling 
    # Y is MxN , UV are both JxK where J,K < M,N 
    # recover pixels so that |J,K| == |M,N|
    #RECOUVER MAY BE THE SOURCE OF BUG

    recouvered_u_matrix = recouver_uv(decompressed_u_pixels, rows, cols)
    recouvered_v_matrix = recouver_uv(decompressed_v_pixels, rows, cols)

    print('recouvered u block: ',recouvered_u_matrix[0:8,0:8])
    print('recouvered v block: ',recouvered_v_matrix[0:8,0:8])

    print('recovered_u_matrix dimensions', recouvered_u_matrix.shape)
    print('recovered_v_matrix dimensions', recouvered_v_matrix.shape)
    print(recouvered_u_matrix)
    
    # 7 - Merge Y,U,V matrices together into a 3D matrix

    yuv_img = yuv_merge(decompressed_y_pixels, recouvered_u_matrix, recouvered_v_matrix, rows, cols)
    print('recouvered v block: ',recouvered_v_matrix[0:8,0:8])

    # 8 - Convert YUV matrix back to RGB
    #converting to rgb MAY BE THE SOURCE OF BUG
    #image_converted_to_rgb = yuv_to_rgb(yuv_img)

    # 9 - Create an Image object from array and render
    decoded_img = Image.fromarray(yuv_img, 'RGB')

    decoded_img.show()



    '''
    FQ_y = DCT(y, rows, cols, DCT_matrix, Q_matrix_LUM)
    FQ_u = DCT(u, uv_rows, uv_cols, DCT_matrix, Q_matrix_CHROME)
    FQ_v = DCT(v, uv_rows, uv_cols, DCT_matrix, Q_matrix_CHROME)

    iFQ_y = iDCT(y, rows, cols, DCT_matrix, Q_matrix_LUM)
    iFQ_u = iDCT(u, uv_rows, uv_cols, DCT_matrix, Q_matrix_CHROME)
    iFQ_v = iDCT(v, uv_rows, uv_cols, DCT_matrix, Q_matrix_CHROME)

    E_y = error(test, FQ_test2, rows, cols)
    E_u = error(test, FQ_test2, uv_rows, uv_cols)
    E_v = error(test, FQ_test2, uv_rows, uv_cols)
    '''


    #R = recover_uv(test, 8, 8)
    #print(R)

    ### CHROMOSUBSAMPLING
    ## How do to the 4:2:0 subsampling

    # img = Image.fromarray(block, 'RGB')
    # print(block)
    # img.show()
    # ss_image.show()







# Checks to see if this is the main module being run
if __name__ == '__main__':
    main()


