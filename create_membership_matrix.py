import pandas as pd
import numpy as np

##Can change values for these 
goalSection = 5 #How much of the guide we use
goalRows = 40
goalCols = 384
##

df = pd.read_excel('MembershipMatrix_Guide.xlsx', header = None, index_col = False)  # change the name of the file as needed

raw_array = df.to_numpy() #Convert to numpy
trans = np.transpose(raw_array) #Take transpose

base8_array = trans[:goalSection]   #Take appropriate section

numRows, numCols = np.shape(base8_array)

binaryArray = np.zeros([goalRows,goalCols], dtype = int)   #Create blank binary array 
print(binaryArray, np.shape(binaryArray))

for row in range(numRows):
    for col in range(numCols):    #Fill in the array, converting to binary  
    
        x = base8_array[row][col]
        startingRow = row*8
        one_placement = startingRow + 7-x
        binaryArray[one_placement][col] = 1

print("final")
print(binaryArray)

df = pd.DataFrame(binaryArray) # convert your array into a dataframe

# save to xlsx file, change name if neccessary
filepath = "membershipMatrix_exp1_" + str(goalRows) + "x" + str(goalCols)+ ".xlsx"

df.to_excel(filepath, index=False)
print("done", filepath)


        




