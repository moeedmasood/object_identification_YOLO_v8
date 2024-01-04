<b> Author: Moeed Masood </b>

This is the third assignment for Quarter 4 of PIAIC class AIC

The objective of this assignment is to identify an object within an image with highest confidence value and crop that object in a separate file.

To run this program, we have following options:

Option 1: type following command:

    python .\assignment3.py

The above command will perform an object identification task using YOLO deep learning convolutional model using a default image. The default image is stored in the folder "image_data" with the name default.jpg

Option 2: select your own picture

    python .\assignment3.py --arg1 .\image_data\meme.jpg

Here I have given a path of an another image called meme.jpg, you can provide the path of your own image here. This will save the cropped output file into the same folder with the prefix "cropped_"

Option 3: select your own picture and also your own path of the output cropped file

    python .\assignment3.py --arg1 .\image_data\meme.jpg --arg2 .\image_data\my_meme_cropped.jpg
