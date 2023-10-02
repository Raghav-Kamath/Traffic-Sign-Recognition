# Testing the trained model
import pandas as pd
import numpy as np
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt
import os


cur_path = os.getcwd()
filename = 'Test\\00484.png' # setting a default filename

model = load_model('./training/TSR.h5') # importing the saved model

print("\nChoose an image file for testing from Test Folder\n")
from tkinter import Tk    
from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename(initialdir = cur_path+"/Test",
										title = "Select an Image File",
										filetypes = (("Image Files",
														"*.png*"),
													("all files",
														"*.*"))) # show an "Open" dialog box and return the path to the selected file
print("The Chosen file is",filename)


# Classes of traffic signs
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

def test_on_img(img):
    data=[]
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = np.argmax(model.predict(X_test), axis=-1)
    return image,Y_pred


plot,prediction = test_on_img(filename)

s = [str(i) for i in prediction] 
a = int("".join(s)) 
print("\nThe Predicted traffic sign is: ", classes[a])
plt.title("Chosen image")
plt.imshow(plot)
plt.show()
