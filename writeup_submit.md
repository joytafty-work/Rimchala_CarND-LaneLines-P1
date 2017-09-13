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
[HSVSelected]: ./report_images/HSV_select.png "HSV selected images"
[HSLSelected]: ./report_images/HLS_select.png "HSL selected images"
[ROISelected]: ./report_images/ROI_select.png "ROI selected images"

---

### Reflection 

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.
#### High-level Summary ####
My pipeline consists of the following steps. 
- [Color Selection](#color-selection)
- [Region of Interest Selection](#region-of-interest-selection)
- [Gray Scaling](#gray-scaling)
- Gaussian Smoothing
- Canny Edge Detection
- Hough Tranform Line Detection

In my experiments, the key to obtaining clean images for lane detection are Color Selection and Region of Interest Selection. The Gray Scaling and Gaussian Smoothing contribute minimally to cleaner images for Edge detection and line detecion. 

#### Color selection ####
The intuition behind color selection is that images taken from a self-driving car dashboard are fairly consistent in their high level composition. For most of the well-paved road, lane lines are painted bright white and yellow against dark gray background (making it obvious for driver to make out the lanes). 

To perform color selection, I experimented with setting range filters for white and yellow colors in RGB, HSV, and HLS color models. I looked up the RGB, HSV, and HSL colors using this [online color picker tool](http://colorizer.org/) and slightly modified the range to get crisper color segmentation. In the test images, with carefully chosen lower and upper bounds for each color models, image patches with white and yellow color cleanly segmented as shown with the HSV and HSL selected test images slightly more cleanly segmented than the RGB selected ones. To choose between HSV and HLS, I have found this [paper](http://revistas.ua.pt/index.php/revdeti/article/viewFile/2092/1964) which compares between HSV, HSL and other color models in real-time objection recognition and found HSV to be the best. So I choose to use HSV selection in my pipeline. Here's my implementation of Color Section. 

```
def hsl_color_select_white_yellow(image):
    # HLS convention : [Hue, Lightness, Saturation]
    # White: Any hue, max lightness, any saturation, 
    # Bright Yellow: Yellow hue (30 - 90), but the yellow in the images can tint toward orange. So I change the lower bound to 10
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    white_mask = cv2.inRange(image, np.array([0, 220, 0]), np.array([255, 255, 255]))
    yellow_mask = cv2.inRange(image, np.array([10, 0, 150]), np.array([90, 240, 240]))
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)
    color_selected = cv2.bitwise_and(image, image, mask = color_mask)
    return cv2.cvtColor(color_selected, cv2.COLOR_HLS2RGB)
```
The test images after the HSV Color Segmentation are shown below: 
##### best HSV color selected outputs #####
![HSV selected images][HSVSelected]

#### Region of Interest Selection ####
The intuition behind region of interest selection is from the car dashboard the bottom half of the image patch are road areas where lane lines are painted. The region of interest selection filters in only the region where it is highly likely for lane lines to be so that the rest of the pipeline focuses on detecting lines from this region.

To implement this, I experimented with the all four corners of the ROI and found that work best for the test images. Here's my implementation: 

```
def compute_roi_vertices(image):    
    image_height, image_width, _ = image.shape
    top_left = [int(0.3*image_width), int(0.45*image_height)]
    top_right = [int(0.7*image_width), int(0.45*image_height)]
    bottom_left = [int(0.05*image_width), 0.95*image_height]
    bottom_right = [int(0.95*image_width), 0.95*image_height]
    
    return np.array([[top_left, top_right, bottom_left, bottom_right]], dtype=np.int32)

roi_selected_images = [
    region_of_interest(image, compute_roi_vertices(image))
    for image in best_color_selected_images
]
```
The HSV selected images after the ROI Selection are shown below:
##### ROI selected outputs #####
![ROI selected images][ROISelected]

#### Gray Scaling ####

### 2. Identify potential shortcomings with your current pipeline

#### Color selection
1. Lighting : All the provided test images are taken from well illuminated scene for which the color segmentation can yield crisp cleanly segmented lane lines from the rest of the scene. In low lighing condition, color segmentation especially based on hue value (for yellow lane line) will be challenging as different color region in low saturation start to overlap. 

2. Lane line coded in different color

3. Rapidly changing illumination (e.g. driving through tunnels)

4. Differently colored lane line or lack of lane lines


### 3. Suggest possible improvements to your pipeline
1. Use previous frames to guess where the next lane line might be
A possible improvement would be to ...

Another potential improvement could be to ...
