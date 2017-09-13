# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[RGBSelected]: ./report_images/RGB_select.png "RGB selected images"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.
My pipeline consists of the following steps. 
- Color Selection
- Region of Interest Selection
- Gray Scaling
- Gaussian Smoothing
- Canny Edge Detection
- Hough Tranform Line Detection

I found that the key to obtaining clean images for lane detection are Color Selection and Region of Interest Selection. Images from a self-driving car dashboard camera angle are fairly consistent in their high level composition. The bottom half of the image patch are road areas where lane lines are. For most of the well-paved road, lane lines are painted bright white and yellow against dark gray background (making it obvious for driver to make out the lanes). 

To perform color selection, we experimented with setting range filters for white and yellow colors in RGB, HSV, and HLS color models. In the test images, with carefully chosen lower and upper bounds for each color models, image patches with white and yellow color cleanly segmented.
http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Changing_ColorSpaces_RGB_HSV_HLS.php

First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![RGB selected images][RGBSelected]

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
