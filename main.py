
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

# Recover UV values using single pass
# WORK ON THIS
def recover_uv(matrix, r, c):
    recover = np.empty([(2 * r), (2 * c)])

    for a in range(0, (2 * r), 1):
        for b in range(0, (2 * c), 1):
            if a % 2 == 0 and b % 2 == 0:    # top left square
                recover[a, b] = matrix[(a // 2)][(b // 2)]

            elif a % 2 == 1 and b % 2 == 1:  # bottom right square
                if b == ((2 * r) - 1) and a == ((2 * r) - 1):   # last row, last column
                    recover[a, b] = matrix[(a // 2) - 1][((b // 2) - 1)]
                elif b == ((2 * r) - 1):  # last column, odd row
                    recover[a, b] = np.mean(matrix[(a // 2) - 1][((b // 2) - 1)] + matrix[(a // 2) + 1][((b // 2) - 1)])
                elif a == ((2 * r) - 1):  # last row, odd column
                    recover[a, b] = np.mean(matrix[(a // 2) - 1][((b // 2) - 1)] + matrix[(a // 2) - 1][((b // 2) + 1)])
                else:
                    recover[a, b] = np.mean(np.mean(matrix[((a // 2) - 1)][((b // 2) - 1)] + matrix[((a // 2) + 1)][((b // 2) + 1)])
                                        + np.mean(matrix[((a // 2) + 1)][((b // 2) - 1)] + matrix[((a // 2) - 1)][((b // 2) + 1)]))

            elif a % 2 == 0 and b % 2 == 1:   # bottom left square
                if a == ((2 * r) - 1):  # last row, even column
                    recover[a, b] = matrix[(a // 2) - 1][(b // 2)]
                else:
                    recover[a, b] = np.mean(matrix[((a // 2) - 1)][(b // 2)] + matrix[((a // 2) + 1)][b // 2])

            elif a % 2 == 1 and b % 2 == 0:   # top right square
                if b == ((2 * r) - 1):  # last column, even row
                    recover[a, b] = matrix[(a // 2)][((b // 2) - 1)]
                else:
                    recover[a, b] = np.mean(matrix[(a // 2)][((b // 2) - 1)] + matrix[(a // 2)][((b // 2) + 1)])

    return recover




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


def DCT(comp, r, c, T, Q):
    # initialize T transpose from T
    TT = np.transpose(T)

    # create new array for DCT'd values
    Fcomp = np.empty([r, c])
    FFcomp = np.empty([r, c])
    FQcomp = np.empty([r, c])

    # iterate through comp in 8x8 blocks
    for i in range(0, r-7, 8):
        for j in range(0, c-7, 8):
            block = comp[i:i+8][j:j+8]

            # apply DCT algorithm to 8x8 block
            Fblock = np.zeros((8, 8),  dtype=np.float64)
            FFblock = np.zeros((8, 8), dtype=np.float64)
            for c in range(0, 8, 1):
                for a in range(0, 8, 1):
                    for b in range(0, 8, 1):
                        # multiply T and block
                        Fblock[c, b] += (T[c, a] * (block[a, b]-128))

            for c in range(0, 8, 1):
                for a in range(0, 8, 1):
                    for b in range(0, 8, 1):
                        # multiply T and block
                        FFblock[c, b] += (Fblock[c, a] * TT[a, b])

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

    return FQcomp


def inverse_DCT(comp, r, c, T, Q):
    # initialize T transpose from T
    TT = np.transpose(T)

    # create new array for iDCT'd values
    iFcomp = np.empty([r, c])
    iFFcomp = np.empty([r, c])
    iFQcomp = np.empty([r, c])

    # iterate through comp in 8x8 blocks
    for i in range(0, r-7, 8):
        for j in range(0, c-7, 8):
            iblock = comp[i:i+8][j:j+8]

            # Apply inverse Quantization to block
            iQblock = np.empty([8, 8])
            for k in range(0, 8, 1):
                for L in range(0, 8, 1):
                    iQblock[k, L] = round(iblock[k, L] * Q[k, L])

            # add block to iFQcomp matrix
            iFQcomp[i:i + 8][j:j + 8] = iQblock

            # apply iDCT algorithm to 8x8 block
            iFblock = np.zeros((8, 8),  dtype=np.float64)
            iFFblock = np.zeros((8, 8), dtype=np.float64)
            for c in range(0, 8, 1):
                for a in range(0, 8, 1):
                    for b in range(0, 8, 1):
                        # multiply T and block
                        iFblock[c, b] += (TT[c, a] * (iQblock[a, b]))

            for c in range(0, 8, 1):
                for a in range(0, 8, 1):
                    for b in range(0, 8, 1):
                        # multiply T and block
                        iFFblock[c, b] += (iFblock[c, a] * T[a, b])

            # add block to iFcomp matrix
            iFcomp[i:i + 8][j:j + 8] = iFFblock

            for a in range(0, 8, 1):
                for b in range(0, 8, 1):
                    iFcomp[a, b] = round(iFcomp[a, b] + 128)

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
    quant_num = 1
    Q_matrix_LUM = initialize_Q_LUM(quant_num)
    Q_matrix_CHROME = initialize_Q_CHROME(quant_num)

    FQ_test = DCT(test, 8, 8, DCT_matrix, Q_matrix_LUM)
    FQ_test2 = inverse_DCT(FQ_test, 8, 8, DCT_matrix, Q_matrix_LUM)
    print(FQ_test)
    print(FQ_test2)

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


    R = recover_uv(test, 8, 8)
    print(R)

    ### CHROMOSUBSAMPLING
    ## How do to the 4:2:0 subsampling

    # img = Image.fromarray(block, 'RGB')
    # print(block)
    # img.show()
    # ss_image.show()







# Checks to see if this is the main module being run
if __name__ == '__main__':
    main()


