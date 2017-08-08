# **Advanced Lane Finding Project**
### *by Maxim Taralov*
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_calibration.png "Undistorted"
[distorted]: ./test_images/test2.jpg "Road Transformed"
[undistorted]: ./output_images/test2_undist.jpg "Road Transformed"
[test5_binary]: ./output_images/coloring.png "Binary Example"
[warping]: ./output_images/warping.png "Warp Example"
[fitlines]: ./output_images/project_top_down.jpg "Fit Visual"
[processed]: ./output_images/processed_frame.jpg "Output"
[processed_challenge]: ./output_images/processed_challenge.jpg "Output"
[video1]: ./output_images/project_out.mp4 "Video"
[video2]: ./output_images/challenge_out.mp4 "Challenge Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "project.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  Since some of the provided images have a 9x6 visible chessboard and some only have 9x5 visible tiles, some of the objpoints are (9x6) and some are (9x5). I also used the OpenCV function `cornerSubPix()` to increase the accuracy of the calibration. I had to erase parts of some of the (9x5) images, so that the algorithm does not get confused.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

The calibration parameters are then saved in the file `camera_info`. If this file exists, the parameters are read instead of computed.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][distorted]

After correction it looks like this:

![alt text][undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for the thresholding is in code cell 5. The function `color_mask()` expects an image after perspective transform (described in the next section). I use color thresholds to generate a binary image. In particular, I convert the warped RGB image in the Luv image space. Then, depending on the number of very bright pixels (in the case of the supplied videos this is an indicator of higher contrast) and the total luminance of the road, I do different things:

1. Low contrast, medium luminance -- I threshold on the L channel to capture white lines and on the v channel to capture yellow lines: `L>190` `v>145`
2. Low contrast, low or high luminance -- The same as the previous point, but different threshold value for the v channel: `L>190` `v>150`
3. High contrast -- In this case, I use a sliding window approach, where in each window, I use histogram equalization on the L and v channel from OpenCV (equalizeHist function) and then threshold on these values. This allows me to also capture detail on lanes partially occluded by shadows. The values are `v_equ_window>247` `L_equ_window>252`

The results are visualized with the following images

![alt text][test5_binary]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_warp_mat()`, which appears in the 6th code cell of the IPython notebook.  The `get_warp_mat()` function finds a trapezoid in a specified by the user vertical window (in my case between `y_min = 480` and `y_max = img_size[0]`). The image is color thresholded and then the four points are found as the innermost points on the top and bottom y-values that are common between the left and the right sides of the image. The function also accepts x-axis adjustments on the found four points for fine-tuning. With minor adjustments, I obtained the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 272, 689      | 300, 715        |
| 1050, 689     | 980, 715      |
| 732, 480      | 980, 400      |
| 556, 480      | 300, 400        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][warping]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To find the points of the lane lines, I use two functions `find_lane_points_reset()` and `find_lane_points_update()` (code cell 9). The first function uses a sliding window approach. In each vertical slice of the window, it finds the maximum mass of points by summing in `y` and if sufficiently many points are found recenters on that `x` value. After that, it selects the nearby points. Before the recentering, however, it checks that the window is not too noisy by comparing the median of the summed values against some threshold. If the window is deemed noisy, the points are rejected.

The second function, `find_lane_points_update()`, simply uses a previous fit and selects the points that are nearby.

If a good fit is not found on the previous frame, the code uses `find_lane_points_reset()`, otherwise it uses `find_lane_points_update()`. After finding the lane points, a trial second order polynomial fit is established using `numpy.polyfit(y,x,deg=2)`.

This fit is then supplied to the `evaluate_fits()` function. There I check the following

* Is the distance between the top pixels of the current fits very different than on the established fits, i.e.
```python
top_diff_prev = right_fit_prev(y_top) - left_fit_prev(y_top)
top_diff_curr = right_fit_curr(y_top) - left_fit_curr(y_top)
abs(top_diff_curr - top_diff_prev) < 120
```
* Analogous check for `y_bottom = 719`
* I also check that the `x` values at the top pixels didn't move too much

If these checks pass, I do a weighted average between the current fit and the previous fits.

```python
best_fit = 0.1*best_fit+0.9*current_fit
```

The line information along with the fits is stored in variables of `class Line`.

The processing of the images is mostly wrapped in `class Processor`, to escape as much as possible from global variables and functions with too much parameters. Upon initialization, one must supply the camera matrix, the distortion coefficients, the transformation and inverse transformation matrices, the pixel to meters conversion ratios and possible correction for the off-center calculations. That last parameter is there, in case we don't have a frame with straight lines, and the car in the center of the lane for computing the transformation matrix. For example, this was the case with the video `challenge_video.mp4`. There, I used a frame where the car was to the left of the center. Also, for that video the warp matrix was different than for `project_video.mp4`.

An example of point selection and fitting of the lanes is

![alt text][fitlines]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculate the curvature with the following short function

```python
def find_curvature(fit, y):
    dxdy2 = np.polyval(np.polyder(fit, 2), y)
    dxdy = np.polyval(np.polyder(fit, 1), y)
    return (1+dxdy**2)**1.5/np.abs(dxdy2)
```

It is defined in the 11th cell of the IPython notebook. Note, that the fit I supply to this function is already fitted on transformed coordinates (in the function `Processor.fit_in_world_coords()`), i.e. this is not the fit I use in the other calculations.

The position is calculated using the following code

```python
  base_left = np.polyval(self.left_line.best_fit, 719)
  base_right = np.polyval(self.right_line.best_fit, 719)
  off_center = (base_right+base_left)*0.5 - image.shape[1]//2 + self.off_center_correction
  ...
  off_center = np.abs(off_center)*self.xm_per_pix
```

Depending on the sign of `off_center` after line 3, the designation is left or right.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The pipeline is implemented in the class `Processor` in code cell 12. Here is an example of my processing of a test image:

![alt text][processed]

And this is from `challenge_video.mp4`

![alt text][processed_challenge]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

This is a link to the processed [project video](./output_images/project_out.mp4) and this to the [challenge video](./output_images/challenge_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

During the course of this project, I tried some of the following methods/techniques

1. For detection I tried thresholding in `HLS` color space, but `Luv` gave me more robust results. I also tried using a `Sobel` kernel to find the gradients in the image, but they were too noisy, so I didn't pursue that approach. Especially, in the challenge video, the shadows and the dark patches on the road were too pronounced. Combined with other filtering (for example luminance for shadow rejection), it might still work.
2. For the fitting of the points, I tried also third order polynomials and `RANSAC` regression from the `sklearn` library. The third order gave better results than the second order in some places, but in others the curves were too strong, so the fits went outside the lanes. RANSAC gave too much stochastic noise and in the end, second order features with RANSAC gave worse results than plain second order polynomials. I didn't have time to try some form of spline interpolation (e.g. B-Splines).
3. For line rejection, I tried using curvature; second derivative of fits (which tells us roughly how strongly convex/concave the curve is); checking if the center of curvature of the fits is the same; checking if the lines are roughly parallel by computing the normal vectors at several points along one line and checking, if the distance between the two lines is approximately the same at these points between the two lines (it should be, if the lines are indeed parallel). It turns out, however, that perspective transform does not provide very stable results (this can be observed, if one watches the project video from top-down view). For example, at several places there are bumps on the road and consequently the transformed lines separate from one another on the way down the bump and then come together on the way up. In the end these methods had too many false positives and led to worse detection, at least in my case. This is the reason I chose line rejection based on the differences between positions of the curves at the highest and lowest pixels.
4. I tried to smooth the fits by using previous fits. In the end I keep a running weighted average with a heavy bias towards the current fit (9:1 ratio). I tried to combine a predicted fit based on interpolation of several previous fits and the current fit, but it didn't work as well. Also lower ratios than the current one led to worse predictions, e.g. it takes some time to settle on the real lane after a bump on the road, rather than instantaneous.

Right now, the most likely place the pipeline can fail is the thresholding. It is relatively fragile and differing conditions can lead to failure to detect. On the most extreme end, if all the pixels on the screen are detected all the time, the lines will never change as there will always be enough points to build the same lines over and over. From a more practical side, I use in several functions `numpy.nonzero()` to select the nonzero elements in a binary image and then do some processing on these pixels. However, the cost of at least some of these functions varies with the number of found points. This means, that potentially something that was real-time can suddenly lag behind.

For the future, I would first investigate rejection in the point cloud. We should compute some statistics such as the standard deviation, skewness and kurtosis of the `x` values (and more specifically in the Euclidian distance between the line and the points) that would allow us to gauge the spread and steepness of the data and compare if these are realistic against known values. Right now, I only do a rudimentary check on the median. Another thing would be to try and make the point detection more robust by adjusting for differing conditions on a full sliding window search (not only vertical slices). This can improve difficult lighting conditions such as tree shadows and tunnels.
