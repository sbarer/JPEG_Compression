import numpy as np




small = np.zeros((192,288), dtype=np.uint8)

for i in range(0,192):
    for j in range(0,288):
        small[i][j] = i *j +5
large = np.zeros((394,590), dtype=np.uint8)

#array[rows, cols]
# rows, cols = array.shape
def recouver(small, lrows,lcols):
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
            print(i,j)
            if (i % 2 ==0 and j %2 ==0): 
                pass
            else:    
                if(j==int((last_col))):   #if the current element is at the very last colum
                    large[i,j] = large[i,j-1]
                else:
                    print(i,j)
                    avg_val = (large[i,j-1] + large[i, j+1])/2  #avg val of two columns
                    avg_val = round(avg_val)
                    large[i,j] = avg_val

    print(large)
                
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



    print('small', small)
    print('large', large)
    print(rows, cols)
    return large
    


recovered_array = recouver(small, 394, 590)
print(recovered_array.shape)
print(recovered_array[0:300,0:500])