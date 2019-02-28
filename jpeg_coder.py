from PIL import Image
import numpy.matlib
import numpy as np
import math
import inspect, os
# To grab arguments from input

class jpeg_coder:
    def __init__(self, compression_val):
        self.compression_val = compression_val

    def file_inputs(self):

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
    def rgb_to_yuv(self,file_path):
        img = Image.open(file_path)
        img.show()
        pixels = img.load()  # Create Pixel map

        width = img.size[0]
        height = img.size[1]

        # RESIZE IMAGE IF IT IS not a scalar multiple of 8x8

        if (width % 8 != 0 or height % 8 !=0):
            # Splice original image
            width_padding = width%8
            height_padding = height%8
            allpix = np.array(img, dtype=np.uint8)
            temp = allpix[0:height-height_padding:1, 0:width-width_padding:1]
            # load spliced image into pixeldata
            img = Image.fromarray(temp, 'RGB')
            pixels = img.load()
            width = img.size[0]
            height = img.size[1]

        # RGB -> YUV conversion
        for i in range(width):
            for j in range(height):
                r = pixels[i, j][0]
                g = pixels[i, j][1]
                b = pixels[i, j][2]
                #y = (0.299*r) + (0.587*g) + (0.114*b)
                #u = 0.492* (b-y)
                #v = 0.877 * (r-y)
                #u = (-0.299*r) - (0.587*g) + (0.886*b) 
                #v = (0.701* r) - (0.587*g) - (0.114*b)
                y = r *  .299000 + g *  .587000 + b *  .114000
                u = r * -.168736 + g * -.331264 + b *  .500000 + 128
                v = r *  .500000 + g * -.418688 + b * -.081312 + 128
                pixels[i,j] = (int(y),int(u),int(v))
        #img.show()
        return img

    def recover_uv(self, matrix, r, c):
    # r & c are the dimensions of the Y matrix, as u&v may have been truncated

        recover = np.empty([r, c])

        for a in range(0, r, 2):
            for b in range(0, c, 1):
                if a % 2 == 0 and b % 2 == 0:  # top left square
                    recover[a][b] = matrix[(a // 2)][(b // 2)]

                elif a % 2 == 0 and b % 2 == 1:  # top right square
                    if b == (r - 1):  # last column
                        recover[a][b] = recover[a, (b - 1)]
                    else:
                        recover[a][b] = (recover[a][(b - 1)] + matrix[(a // 2)][((b // 2) + 1)]) // 2

        for a in range(1, r, 2):
            for b in range(0, c, 1):
                if a % 2 == 1 and b % 2 == 0:  # bottom left square
                    if a == (r - 1):  # last row, even column
                        recover[a][b] = recover[(a - 1)][b]
                    else:
                        recover[a][b] = (recover[(a - 1)][b] + recover[(a + 1)][b]) // 2

                elif a % 2 == 1 and b % 2 == 1:  # bottom right square
                    if b == (r - 1) and a == (r - 1):  # last row, last column
                        recover[a][b] = recover[(a - 1)][(b - 1)]
                    elif b == (r - 1):  # last column
                        recover[a][b] = (recover[(a - 1)][(b - 1)] + recover[(a + 1)][(b - 1)]) // 2
                    elif a == (r - 1):  # last row
                        recover[a][b] = (recover[(a - 1)][(b - 1)] + recover[(a - 1)][(b + 1)]) // 2
                    else:
                        recover[a][b] = (((recover[(a - 1)][(b - 1)] + recover[(a + 1)][(b + 1)]) // 2)
                                        + (recover[(a + 1)][(b - 1)] + recover[(a - 1)][(b + 1)]) // 2) // 2

        return recover

    def recover_uv_v2(self, matrix, r, c):
    # r & c are the dimensions of the Y matrix, as u&v may have been truncated
        recover = np.empty([r, c])

        # calculate size of new temp array
        size_r = r // 2
        size_c = c // 2
        last_col = c - 1
        last_row = r - 1
        temp = np.zeros((size_r, size_c))

        # calculate amount of padding to add
        mat_rows, mat_cols = matrix.shape
        pad_r = size_r - mat_rows
        pad_c = size_c - mat_cols

        # account for all 4 possibilities of matrix resizing
        if pad_r != 0 and pad_c != 0:
            temp[:-pad_r, :-pad_c] = matrix
        elif pad_r != 0:
            temp[:-pad_r, :] = matrix
        elif pad_c != 0:
            temp[:, :-pad_c] = matrix
        else:
            temp = matrix

        # fill in padded columns
        for i in range(mat_cols, size_c, 1):
            for j in range(size_r):
                temp[j, i] = temp[j, (mat_cols-1)]

        # fill in padded rows
        for i in range(mat_cols):
            for j in range(mat_rows, size_r, 1):
                temp[j, i] = temp[(mat_rows-1), i]


        for a in range(0, r, 2):
            for b in range(c):
                if a % 2 == 0 and b % 2 == 0:  # top left square
                    recover[a][b] = temp[(a // 2)][(b // 2)]

                elif a % 2 == 0 and b % 2 == 1:  # top right square
                    if b == last_col:  # last column
                        recover[a][b] = recover[a, (b - 1)]
                    else:
                        recover[a][b] = (recover[a][(b - 1)] + temp[(a // 2)][((b // 2) + 1)]) // 2


        for a in range(1, r, 2):
            for b in range(c):
                if a % 2 == 1 and b % 2 == 0:  # bottom left square
                    if a == last_row:  # last row, even column
                        recover[a][b] = recover[(a - 1)][b]
                    else:
                        recover[a][b] = (recover[(a - 1)][b] + recover[(a + 1)][b]) // 2

                elif a % 2 == 1 and b % 2 == 1:  # bottom right square
                    if b == last_col and a == last_row:  # last row, last column
                        recover[a][b] = recover[(a - 1)][(b - 1)]
                    elif b == last_col:  # last column
                        recover[a][b] = (recover[(a - 1)][(b - 1)] + recover[(a + 1)][(b - 1)]) // 2
                    elif a == last_row:  # last row
                        recover[a][b] = (recover[(a - 1)][(b - 1)] + recover[(a - 1)][(b + 1)]) // 2
                    else:
                        recover[a][b] = (((recover[(a - 1)][(b - 1)] + recover[(a + 1)][(b + 1)]) // 2)
                                        + (recover[(a + 1)][(b - 1)] + recover[(a - 1)][(b + 1)]) // 2) // 2


        return recover

    # Recouver UV values using single pass
    # WORK ON THIS
    #array[rows, cols]
    # rows, cols = array.shape
    def recouver_uv(self, small, lrows,lcols):
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
                        avg_val = (large[i,j-1])/2 + (large[i, j+1])/2  #avg val of two columns
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
                    avg_val = (large[i-1,j])/2 + (large[i+1,j])/2
                    avg_val = round(avg_val)
                    large[i,j] = avg_val

        #input small array values into large array 



        #print('small', small)
        #print('large', large)
        #print(rows, cols)
        return large
        


    def yuv_merge(self, y, u, v, rows, cols):
        merged_yuv = np.zeros((rows, cols,3), dtype=np.float64)
        for i in range(0,rows):
            for j in range(0,cols):
                merged_yuv[i,j][0] = int(y[i,j])
                merged_yuv[i,j][1] = int(u[i,j])
                merged_yuv[i,j][2] = int(v[i,j])

        return merged_yuv


    def yuv_to_rgb(self, matrix):
        rgb_converter = np.array([[1.0, 1.0, 1.0],
                    [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                    [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])

        rgb_matrix = np.dot(matrix, rgb_converter)
        rgb_matrix[:, :, 0] -= 179.45477266423404
        rgb_matrix[:, :, 1] += 135.45870971679688
        rgb_matrix[:, :, 2] -= 226.8183044444304

        return rgb_matrix
        
    #TODO: write the converter
    #TODO: write the converter
    def yuv_to_rgb2(self, matrix):
        row, col, depth = matrix.shape
        #img = Image.fromarray(matrix, 'RGB')
        #img.show()
        rgb_matrix = np.zeros((row,col,depth), dtype=np.uint8)
        for i in range(row):
            for j in range(col): 
                y = matrix[i,j][0]
                u = matrix[i,j][1]
                v = matrix[i,j][2]
                
                #if((v + y) > 255):
                #    print('r is bigger than 255')
                ##    r = 255
                #else:
                #    r = v + y 

                #if((u + y) > 255):
                #    print('b is bigger than 255')
                #    b = 255
                #else:
                #    b = u + y
                #if((y -(0.299*r) - (0.114*b))/0.587 > 255):
                #    print('g is bigger than 255')
                #    print((y -(0.299*r) - (0.114*b))/0.587)
                #    g = 255
                #else:
                 #   g = (y -(0.299*r) - (0.114*b))/0.587
                r = y + 1.4075 * (v - 128)
                g = y - 0.3455 * (u - 128) - (0.7169 * (v - 128))
                b = y + 1.7790 * (u - 128)
                

                rgb_matrix[i,j][0] = int(r)
                rgb_matrix[i,j][1] = int(g)
                rgb_matrix[i,j][2] = int(b)
                
        return rgb_matrix

    # applys a 4:2:0 chroma subsampling to image
    # returns 3 2D arrays corresponding to YUV
    # FORMAT: pixels[width, height] np.array[height, width]
    # TODO: Try taking average
    def chroma_ss_process(self, img):
        width = img.size[0]
        height = img.size[1]
        pixels = img.load()
        # find chroma block size
        y = np.zeros((height, width), dtype=np.uint8)
        h = height // 2
        w = width // 2
        u = np.zeros((h, w), dtype=np.uint8)
        v = np.zeros((h, w), dtype=np.uint8)
        u_test = np.zeros((height, width), dtype=np.uint8)
        v_test = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                if (i % 2 == 0 and j % 2==0):
                    y[i, j] = pixels[j, i][0]
                    u[i//2, j//2] = pixels[j, i][1]
                    v[i//2, j//2] = pixels[j, i][2]
                    u_test[i, j] = pixels[j, i][1]
                    v_test[i, j] = pixels[j, i][2]
                else:
                    u_test[i, j] = pixels[j, i][1]
                    v_test[i, j] = pixels[j, i][2]
                    y[i, j] = pixels[j, i][0]

        # trim padding to make array a multiple of 8
        u_height, u_width = u.shape[0], u.shape[1]
        v_height, v_width = v.shape[0], v.shape[1]

        if(u_width % 8 != 0 or u_height % 8 != 0):
            width_padding = u_width % 8
            height_padding = u_height % 8
            u = u[0:u_height - height_padding:1, 0:u_width - width_padding:1]
            v = v[0:v_height - height_padding:1, 0:v_width - width_padding:1]

        test_y = Image.fromarray(y)
        test_y.show()
        test_u = Image.fromarray(u_test)
        test_u.show()
        test_v = Image.fromarray(v_test)
        test_v.show()

        #print('u', u)
        #print('dimensions', u.shape)
        return y, u, v


    def initialize_DCT_matrix(self):
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
    def apply_DCT(self, pixels, r, c, DCT_matrix, Q):
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
                        block[x,y] = pixels[i+x,j+y]
                

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



    def inverse_DCT(self,pixels, rows, cols, DCT_matrix, Q):
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


    def initialize_Q_LUM(self, Q_fact):
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


    def initialize_Q_CHROME(self, Q_fact):
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


    def error(self, original, compressed, r, c):
        Error = np.empty([r, c])

        for a in range(0, r, 1):
            for b in range(0, c, 1):
                Error[a, b] = (original[a, b] - compressed[a, b])

        return Error


    def encode(self, file_path):
        '''
        np.set_printoptions(threshold=np.inf)
        input , output = file_inputs()
        print("input file name = " + str(input))
        print("output file name = " + str(output))
        '''

        # 1 - convert RGB to YUV, resize to fit 8x8
        
        image = self.rgb_to_yuv(file_path)
        before_comp_file_size = np.array(image).nbytes
        # 2 - Split YUV into Y U V and subsequent downsampling/ grab reduced size
        y, u, v = self.chroma_ss_process(image)
        y_bytes = y.nbytes
        u_bytes = u.nbytes
        v_bytes = v.nbytes
        after_comp_file_size = y_bytes + u_bytes + v_bytes


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
        DCT_matrix = self.initialize_DCT_matrix()
        quant_num = self.compression_val
        Q_matrix_LUM = self.initialize_Q_LUM(quant_num)
        Q_matrix_CHROME = self.initialize_Q_CHROME(quant_num)

        compressed_y_pixels = self.apply_DCT(y, rows, cols, DCT_matrix, Q_matrix_LUM)
        compressed_u_pixels = self.apply_DCT(u, uv_rows, uv_cols, DCT_matrix, Q_matrix_CHROME)
        compressed_v_pixels = self.apply_DCT(v, uv_rows, uv_cols, DCT_matrix, Q_matrix_CHROME)

        print('compressed y block: ',compressed_y_pixels[0:8,0:8])
        # 5 - Decode Pixel values 
        decompressed_y_pixels = self.inverse_DCT(compressed_y_pixels, rows, cols, DCT_matrix, Q_matrix_LUM)
        decompressed_u_pixels = self.inverse_DCT(compressed_u_pixels, uv_rows, uv_cols, DCT_matrix, Q_matrix_CHROME)
        decompressed_v_pixels = self.inverse_DCT(compressed_v_pixels, uv_rows, uv_cols, DCT_matrix, Q_matrix_CHROME)
        print('decompressed y block: ',decompressed_y_pixels[0:8,0:8])
            
        # 6 - Recover UV pixels lost from chroma subsampling 
        # Y is MxN , UV are both JxK where J,K < M,N 
        # recover pixels so that |J,K| == |M,N|
        #RECOUVER MAY BE THE SOURCE OF BUG

        recouvered_u_matrix = self.recover_uv_v2(decompressed_u_pixels, rows, cols)
        recouvered_v_matrix = self.recover_uv_v2(decompressed_v_pixels, rows, cols)

        recovered_u = Image.fromarray(recouvered_u_matrix)
        recovered_u.show()
        recovered_v = Image.fromarray(recouvered_v_matrix)
        recovered_v.show()

        print('recouvered u block: ',recouvered_u_matrix[0:8,0:8])
        print('recouvered v block: ',recouvered_v_matrix[0:8,0:8])

        print('recovered_u_matrix dimensions', recouvered_u_matrix.shape)
        print('recovered_v_matrix dimensions', recouvered_v_matrix.shape)
        print(recouvered_u_matrix)
        
        # 7 - Merge Y,U,V matrices together into a 3D matrix

        yuv_img = self.yuv_merge(decompressed_y_pixels, recouvered_u_matrix, recouvered_v_matrix, rows, cols)
        print('recouvered v block: ',recouvered_v_matrix[0:8,0:8])

        # 8 - Convert YUV matrix back to RGB
        #converting to rgb MAY BE THE SOURCE OF BUG
        image_converted_to_rgb = self.yuv_to_rgb2(yuv_img)

        # 9 - Create an Image object from array and render
        decoded_img = Image.fromarray(image_converted_to_rgb, 'RGB')

        decoded_img.show()

        # 10 - Save image in assets folder to render on React Page
        current_dir =  os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
        image_subdir = "src/assets/images"
        compression_val = int(self.compression_val * 10)
        fileList = file_path.split('./src/assets/images/')
        fileName = fileList[1].split('.')
        subdir_file_path = fileName[0]
        file_name = str(subdir_file_path) + str(compression_val) + '.jpg'
        image_dir = os.path.join(current_dir, image_subdir, file_name)
    
         ## SAVE Image into file path
        decoded_img.save(image_dir) 

        #GET SIZE OF IMAGE IN BYTES
        
        print(image_dir)
        
        print('before compression file size', before_comp_file_size/1000)
        print('after compression file size',after_comp_file_size/1000)

       
        
        return int(before_comp_file_size/1000), int(after_comp_file_size/1000)


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

    jpeg = jpeg_coder(0.1)
    jpeg.encode('./src/assets/images/mountain.png')
    


