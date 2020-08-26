# Air Paint
## Overview
This project uses OpenCV and TensorFlow to attempt to classify the shape you've drawn with an object of your choice in the air as one of the digits 0-9. 

## Usage
Required packages can be installed with 

`pip install -r requirements.txt`

Create the model by running the ``create_model.py`` script and moving it to the parent directory. 

To begin painting, run the ``air_paint.py`` script. Pressing 'c' clears the canvas and 'q' quits and saves the drawing in the Drawings folder.

### "Tracking"
To "track" the object it uses OpenCV to mask each frame based on the mean HSV color values of the object you've selected. Adjusting the number of standard deviations away from this mean can be useful if the object you are painting with is similar in color to the background. By default, the masked video is shown in addition to the canvas on which you are painting. Use this to fine tune the masking color bounds such that only the object you want to paint with is visible. 

### Classification
Currently, I am using a CNN with the LeNet-5 architecture trained on the MNIST dataset to classify the painted digits. It seems to do rather well so I think it is sufficient for the time being. 

## Example
Here is an example of what the canvas looks like as you are drawing. 
[[https://github.com/ryan570/air-paint-digit-classifier/master/Drawings/08-25-2020_22-38.png|alt=example_image]]