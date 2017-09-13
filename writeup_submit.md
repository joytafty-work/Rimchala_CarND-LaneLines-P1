# **Finding Lane Lines on the Road** 

## Table of Contents ##
- [Pipeline Description](#pipeline-description)
- [Experiments on Test Videos](#experiments-on-test-videos)
- [Potential ShortComings](#potential-shortcomings)
- [Suggestion for Improvements](#suggestion-for-improvements)

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
[EdgeMasked]: ./report_images/Canny_masked.png "Canny edge masked images"
[WeightedMasked]: ./report_images/hough_line_masked.png "Hough line detected images"
[ImprovedDrawLineMasked]: ./report_images/improved_draw_lines_masked.png "Hough line images after improved draw line"
[SolidWhiteVideo]: /test_videos_output/solidWhiteRight.mp4 "SolidWhiteRight test output videos"

---

### Pipeline description ###
#### High-level Summary ####
My pipeline consists of the following steps:
- [Color Selection](#color-selection)
- [Region of Interest Selection](#region-of-interest-selection)
- [Gray Scaling](#gray-scaling)
- [Gaussian Smoothing](#gaussian-smoothing)
- [Canny Edge Detection](#canny-edge-detection)
- [Hough Tranform Line Detection](#hough-transform-line-detection)

I modify the `draw_lines` function to make the line detection more robust by grouping detected lines into left and right lane lines and perform line extrapolation within group. Here is more detailed description: 
- [Improvement to the draw_lines function](#improvement-to-the-draw_lines-function)

In my experiments, the key to obtaining clean images for lane detection are Color Selection and Region of Interest Selection. Gaussian Smoothing seem to contributes minimally to cleaner images for Edge detection and line detecion. 

###### back to [Table of Contents](#table-of-contents)

#### Color selection ####
The intuition behind color selection is that images taken from a self-driving car dashboard are fairly consistent in their high level composition. For most of the well-paved road, lane lines are painted bright white and yellow against dark gray background (making it obvious for driver to make out the lanes). 

To perform color selection, I experimented with setting range filters for white and yellow colors in RGB, HSV, and HLS color models. I looked up the RGB, HSV, and HSL colors using this [online color picker tool](http://colorizer.org/) and slightly modified the range to get crisper color segmentation. In the test images, with carefully chosen lower and upper bounds for each color models, image patches with white and yellow color cleanly segmented as shown with the HSV and HSL selected test images slightly more cleanly segmented than the RGB selected ones. To choose between HSV and HLS, I have found this [paper](http://revistas.ua.pt/index.php/revdeti/article/viewFile/2092/1964) which compares between HSV, HSL and other color models in real-time objection recognition and found HSV to be the best. So I choose to use HSV selection in my pipeline. Here's my implementation of Color Section. 

```python
def hsv_color_select_white_yellow(image):
    # HSV convention : [Hue, Saturation, Value]
    # White: any hue, high saturation, any value
    # Bright Yellow: Yellow hue (30 - 90)
    #   - the yellow in the test images some time tints toward orange I change the lower bound to 10
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    white_mask = cv2.inRange(image, np.array([0, 0, 225]), np.array([255, 255, 255]))
    yellow_mask = cv2.inRange(image, np.array([10, 100, 100]), np.array([90, 255, 255]))
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)
    color_selected = cv2.bitwise_and(image, image, mask = color_mask)
    return cv2.cvtColor(color_selected, cv2.COLOR_HSV2RGB)    
```
The test images after the HSV Color Segmentation are shown below: 
##### best HSV color selected outputs #####
![HSV selected images][HSVSelected]
###### back to [Table of Contents](#table-of-contents)

#### Region of Interest Selection ####
The intuition behind region of interest selection arises from that the car dashboard the bottom half of the image patch are road areas where lane lines are painted. The region of interest selection filters in only the region where it is highly likely for lane lines to be so that the rest of the pipeline focuses on detecting lines from this region.

To implement this, I experimented with the all four corners of the ROI and found that the values that work best for the test images. Here's my implementation: 

```python
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
###### back to [Table of Contents](#table-of-contents)

#### Gray Scaling ####
Gray scaling converts the image in color space into a single channel for downstream processing (Gaussian smoothing and edge detection). For this step, I used the provided `gray_scale` function (which is a thin wrapper code around `cv2.cvtColor( _ , cv2.COLOR_RGB2GRAY)`). 

#### Gaussian Smoothing ####
Gaussian smoothing suppresses (salt and pepper type of) noise from the images to prevent unwanted edges in the edge detection step in the pipeline. The key hyperparameter for Gaussian smoothing is `kernal_size` which specifies the size in pixels of the square slide window on which the 2D Gaussian kernel is convolved over the image. Optimal kernel size should be larger than the largest size of noise while smaller than the smallest key feature size from which we want to detect, so the optimal values depends on the quality of the input images. I experimented with `kernel_size` 3, 5, 7, 9, 11 and chose 5 to be the best kernel size for the pipeline based on the visual inspection of the test images. 

#### Canny Edge Detection ####
To perform edge detection, I use the provided `canny` function which is a thin wrapper around `cv2.Canny()`. The function requires two arguments: `lower_threshold` and `upper_threshold` which defines acceptance criteria for an edge. According to openCV [tutorial](http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html), for a given pixel, 
* if its intensity gradient is above the `upper_threshold`, it is an edge. 
* if an edge's intensity gradient is between the `upper_threshold` and the `lower_threshold` then it is an edge if and only if at least one of their neighboring pixel is an edge
* otherwise it is NOT an edge.
The input images in this project are 8-bit (0-255 pixel intensity) so (the absolute value of) the possible gradient range is 0-255. I experiment with adjusting the `upper_threshold` while keeping `lower_threshold=0` to get clean edge images. I found that any `upper_threshold` values between 100 - 200 yielded clean edge images so I choose to implement `upper_threshold=150` in my pipeline. In my pipeline, the `lower_threshold` does not seem to affect the output edge images that much so I choose to implement with `lower_threshold=50`.

The images after the Canny edge detection algorithm is applied are shown below:
##### Canny edge detected outputs #####
![Canny edged masked images][EdgeMasked]
###### back to [Table of Contents](#table-of-contents)

#### Hough Tranform Line Detection ####
The Hough Transform is the key step in successful line detection. I use the provided `hough_lines` which calls `cv2.HoughLinesP` and draw the detect lines over the original image using `draw_line()` There are five parameters that specifies the output of the `cv2.HoughLines`
* rho - the distance resolution in pixels of the Hough grid
* theta - the angular resolution in radians of the Hough grid
* threshold - the minimum number of intersections in Hough grid cell requires for a point in Hough space to be an edge
* min_line_len - the minimum number of pixels requires for an edge segment to be a line
* max_line_gap - the maximum gap in pixels between connectable line segments to be connected as one line
Varying these parameters requires a lot of experimentation. I found that the values provided in the lesson yielded pretty good results and deviations from those values degrade the quality of the detected lines. For the final implementation, I use the following set of values: 
`rho = 1, theta = np.pi/180, threshold = 20, min_line_len = 10, max_line_gap = 100`. 

The detected Hough lines overlaid on top of the original test images are shown below:
##### Hough line detected outputs overlaid on the original test images #####
![Weight masked images][WeightedMasked]
###### back to [Table of Contents](#table-of-contents)

#### Improvement to the draw_lines function ####
I exploit two strategies to make the `draw_lines()` function more robust. The intuition behind the improvements are as follow. Given the the preprocessing yield high quality Hough lines that can be grouped into left vs. right lane lines segments, the outputs of the Hough transform can be grouped based on their slopes. With `matplotlib.image` y-axis convention (which is the reverse of `matplotlib.pyplot`), negative slope lines belong to the left lane and positive slope lines to the right lane. After the lane grouping, the lines can be extrapolated by using a linear equation with average slope and average position of the line segments within group. 

The implementation of these strategies is shown here: 
```python 
def draw_lines_improved(img, lines, 
                        color=[255, 0, 0], # default to red line
                        thickness=3
                       ):
    """
    Improve draw_lines by 
    * Lane grouping: separating line segments as part of the left line vs. the right line.  
    * Extrapolation: averaging the position of each of the lines and extrapolate to the top and bottom of the lane.    
    """
    # Create a pandas data frame of line for easy aggregation
    # lines are defined as a list of starting point and end point coordinates : x1, y1, x2, y2
    lines_df = pd.DataFrame(list(chain(*lines)), columns = ['x1', 'y1', 'x2', 'y2'])
    # Compute line segment slope and determine if line belongs to the left or right lanes
    lines_df['slope'] = lines_df.apply(lambda row: (row['y2'] - row['y1'])/(row['x2'] - row['x1']), axis=1)
    # Negative slope segments belong to the left lane group (image y increasing from top to bottom)
    lines_df['lane'] = lines_df['slope'].apply(lambda x: 'left' if x < 0 else 'right')
    
    image_height, image_width, _ = image.shape
    # Heuristics for top and bottom pixels
    y1_output = int(0.575*image_height)
    y2_output = int(0.975*image_height)
    
    # group by lane type: 'left' vs. 'right'
    # groupby object are tuples of (group_name, rows)
    for gr in lines_df.groupby('lane'):
        avg_x1 = gr[1]['x1'].agg(np.nanmean).values[0]
        avg_y1 = gr[1]['y1'].agg(np.nanmean).values[0]
        avg_slope = gr[1]['slope'].agg(np.nanmean).values[0]
        if avg_slope != 0:
            x1 = int(avg_x1 + (y1_output - avg_y1)/avg_slope)
            x2 = int(avg_x1 + (y2_output - avg_y1)/avg_slope)
            cv2.line(img, (x1, y1_output), (x2, y2_output), color, thickness)
```
The detected Hough lines overlaid on top of the original test images are shown below:
##### Hough line detected after the modification to the `draw_lines()` function #####
![Improved Draw Lines images][ImprovedDrawLineMasked]
###### back to [Table of Contents](#table-of-contents)

### Experiments on Test Videos ###
The implemented pipeline perform relatively well on the two test videos provided after the improvements on the `draw_lines()` function. The results can be viewed here: 
![Solid White Test Video Output][SolidWhiteVideo]

### Potential ShortComings ###
While 
1. Lighting : All the provided test images are taken from well illuminated scene for which the color segmentation can yield crisp cleanly segmented lane lines from the rest of the scene. In low lighing condition, color segmentation especially based on hue value (for yellow lane line) will be challenging as different color region in low saturation start to overlap. 

2. Lane line coded in different color

3. Rapidly changing illumination (e.g. driving through tunnels)

4. Lane line grouping will fail when the car is turning


### Suggestion for Improvements ###
1. Use previous frames to guess where the next lane line might be
A possible improvement would be to ...

Another potential improvement could be to ...

###### back to [Table of Contents](#table-of-contents)

