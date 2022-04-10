from PIL import Image
import numpy as notnp


#functions =========================================================================================
def imgToArr(fileName):


    picture = Image.open("pics/" + fileName)
    final_pic = picture.convert('1')
    arr = notnp.array(final_pic)
    imageArr = []
    row=[0,0,0,0,0]
    
    for h in range(9):
        
        row=[0,0,0,0,0]
        for i in range(5):
            
            if arr[h][i] == False:
                row[i] = 1
                
        imageArr.append(row)
    return imageArr



#main code =========================================================================================

array = imgToArr("7.png")
print(array)



