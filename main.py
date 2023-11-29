import cv2
import numpy as np

from utils import approximate_contours
from utils import corner_detector
from utils import crop_image_percentage_based
from utils import display_image_in_terminal
from utils import draw_contours_in_image
from utils import draw_circles
from utils import draw_labeled_circles
from utils import final_destination_corners
from utils import hierarchical_contour_filtering_by_size
from utils import order_points_clockwise
from utils import perspective_transformation
from utils import pre_process_image
from utils import pre_process_scanned_image

# ------------------------ Document scanner -------------------------------------------------------------------------------------------

'''
This section involves document scanning and perspective correction:

# Read the original image for document scanning
# Pre-process the original image to enhance document features
# Find and sort contours within the pre-processed image by area
# Visualize the original image with the biggest contour detected
# Detect and label corners of the largest contour
# Rearrange and visualize the corners in a clockwise order
# Calculate the final destination corners for perspective correction
# Perform perspective transformation based on the ordered corners
# Crop the transformed image to remove external borders
'''

# Read the original image
image = cv2.imread('IMG_4801.png')
display_image_in_terminal(image, "Original image")

# Pre-process the original image
pre_processed_image = pre_process_image(image)
display_image_in_terminal(pre_processed_image, "Pre-processed image")

# Find all contours of 'pre_processed_image'
contours, _ = cv2.findContours(pre_processed_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# Sort 'contours' by their area in descending order 
sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)

# Find the biggest contour
biggest_contour = sorted_contours[0]
image_with_biggest_contour = image.copy()
draw_contours_in_image(biggest_contour, image_with_biggest_contour)
display_image_in_terminal(image_with_biggest_contour, "Original image with the biggest contour")

# Find the corners of 'biggest_approximated_contour'
corners = corner_detector([biggest_contour], pre_processed_image, 50)
corners = np.array(corners).reshape(-1, 2)

# Rearrange 'corners' to order: top-left, top-right, bottom-right, bottom-left
ordered_corners = order_points_clockwise(corners)
ordered_corners = np.array(ordered_corners)
image_with_ordered_corners = image.copy()
draw_labeled_circles(np.array(ordered_corners), image_with_ordered_corners)
display_image_in_terminal(image_with_ordered_corners, "Original image with ordered corners")

# Calculate the final destination corners for perspective correction
destination_corners = final_destination_corners(ordered_corners)

# Perform perspective correction with destination corners
perspective_transformed_image = perspective_transformation(ordered_corners, destination_corners, image)

# Crop the image to remove external borders
cropped_image = crop_image_percentage_based(perspective_transformed_image, 1)
display_image_in_terminal(cropped_image, "Perspective transformed and cropped image")

# ------------------------ Table's corners detector ------------------------------------------------------------------------------------

'''
This section aims to detect the corners defining the table within the image:

# Pre-process the cropped image to enhance features
# Find and visualize all contours within the pre-processed image
# Identify the largest contour as the external borders of the document
# Filter contours to isolate the upper and lower rectangles assumed to represent table sections
# Approximate these contours to simplify corner detection
# Detect corners within the approximated contours for table corners visualization
'''

# Pre-process the cropped image
pre_processed_image = pre_process_scanned_image(cropped_image)
display_image_in_terminal(pre_processed_image, "Pre-processed image")

# Find all contours in 'pre_processed_image'
contours, hierarchy = cv2.findContours(pre_processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
image_with_contours = cropped_image.copy()
draw_contours_in_image(contours, image_with_contours)
display_image_in_terminal(image_with_contours, "Perspective transformed image with all contours")

# Assuming that the largest contour represents the exterrnal borders of the document, the second largest contour represents the upper
# rectangle and the third largest contour reprsents the lower rectangle

# Filter 'contours' to get the contours that are children of the second largest contour
filtered_contours_1 = hierarchical_contour_filtering_by_size(contours, hierarchy, 2)

# Filter 'contours' to get the contours that are children of the third largest contour
filtered_contours_2 = hierarchical_contour_filtering_by_size(contours, hierarchy, 3)
image_with_filtered_contours = cropped_image.copy()
draw_contours_in_image(filtered_contours_1, image_with_filtered_contours)
draw_contours_in_image(filtered_contours_2, image_with_filtered_contours)
display_image_in_terminal(image_with_filtered_contours, "Perspective transformed image with filtered contours")

# Approximate the contours in 'filtered_contours_1' and 'filtered_contours_2'
approximated_contours_1 = approximate_contours(filtered_contours_1)
approximated_contours_2 = approximate_contours(filtered_contours_2)

# Get the coordinates of the corners of all contours from 'approximated_contours_1' and 'approximated_contours_2'
corners_of_approximated_contours_1 = corner_detector(approximated_contours_1, pre_processed_image, 4)
corners_of_approximated_contours_2 = corner_detector(approximated_contours_2, pre_processed_image, 4)
image_with_corners = cropped_image.copy()
draw_circles(corners_of_approximated_contours_1, image_with_corners)
draw_circles(corners_of_approximated_contours_2, image_with_corners)
display_image_in_terminal(image_with_corners, "Perspective transformed image with corners of the filtered contours")