Color quantization Steps: file Cartoon_ID.py
1.	Bilateral filter: First, I implemented a Bilateral filter on input image, filtering is important to perform image processing as it remove noise from the given image. Apart from smoothing the image bilateral filter also preserves the edges of the image. It uses both gaussian and range filter to do that.
2.	Colour Quantization: In this technique colours of an image are grouped in different regions as per the RGB and then replace them wiUth colour which represent the region. To perform this Uniform quantization has been used. I divided the image colour in four regions [0-63], [64,127], [128,191], [191,255].
a.	Divide RGB component in equal parts.
b.	For every pixel in image assign it to its RGB region.
c.	Determine the colour by taking the mean of all values in that region.
d.	Iterate over the image pixels and change the colour by representative of that region.

Circle detection: CountShapes_ID.py
Parametric equation of a circle :
X = a+r*cos(t)
Y = b+r*sin(t) where t∈[0,2π), r is radius and (a,b) are centre.
Steps for Implementation:
1.	All possible circles are defined by values of a,b,r as per the parametric equation of circle.
2.	Pixels for each circle can be determined by looping over t (steps). Only those circles will be detected which are between rmin (minimum radius) and rmax (maximum radius). Taking higher value of radius makes the system computationally expensive.
3.	First convert the image to grayscale and through canny edge detector create an edge of the input image.
4.	Use Hough transformation to determine the centre of the circles. Centre will be chosen where sin and cos value intersects each other.
5.	Once edge is created, iterate over only strong pixels, and find the coordinates of centre of the circle that pass by the point (as per the parametric equation).
6.	Set a threshold of 40% for circles which are good enough. We need to make sure that circle with common centre are not rejected.
7.	Finally to detect or draw the circle, create an empty matrix of the shape of the image and pass the coordinates and radius to skimage.draw.circle_perimeter and assign each pixel a value of 1.


Install following libraries:
- Numpy
- skimage
- matplotlib
- collections

Color Quantization:

Input:

![image](https://user-images.githubusercontent.com/43110863/123035004-f2f9b500-d42d-11eb-98e4-8a6f1f194471.png)

Output:

![image](https://user-images.githubusercontent.com/43110863/123035394-96e36080-d42e-11eb-9b3d-a0741ef58d7c.png)

Circle Detection:

Input:

![image](https://user-images.githubusercontent.com/43110863/123035452-b11d3e80-d42e-11eb-8bb7-88c184ed7f6d.png)

Output:

![image](https://user-images.githubusercontent.com/43110863/123035482-bd090080-d42e-11eb-8324-39f44f3224e1.png)
