import cv2
from utils_text_detector import approximate_contours
from utils_text_detector import corner_detector
from utils_text_detector import display_image_in_terminal
from utils_text_detector import draw_circles
from utils_text_detector import draw_contours_in_image
from utils_text_detector import hierarchical_contour_filtering_by_size
from utils_text_detector import pre_process_scanned_image

'''
This section aims to detect the corners defining the table within the image:

# Pre-process the image to enhance features
# Find and visualize all contours within the pre-processed image
# Identify the largest contour as the external borders of the document
# Filter contours to isolate the upper and lower rectangles assumed to represent table sections
# Approximate these contours to simplify corner detection
# Detect corners within the approximated contours for table corners visualization
'''

def text_detector(image):
    # Pre-process the cropped image
    pre_processed_image = pre_process_scanned_image(image)
    display_image_in_terminal(pre_processed_image, "Pre-processed image")

    # Find all contours in 'pre_processed_image'
    contours, hierarchy = cv2.findContours(pre_processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    image_with_contours = image.copy()
    draw_contours_in_image(contours, image_with_contours)
    display_image_in_terminal(image_with_contours, "Perspective transformed image with all contours")

    # Assuming that the largest contour represents the exterrnal borders of the document, the second largest contour represents the upper
    # rectangle and the third largest contour reprsents the lower rectangle

    # Filter 'contours' to get the contours that are children of the second largest contour
    filtered_contours_1 = hierarchical_contour_filtering_by_size(contours, hierarchy, 2)

    # Filter 'contours' to get the contours that are children of the third largest contour
    filtered_contours_2 = hierarchical_contour_filtering_by_size(contours, hierarchy, 3)
    image_with_filtered_contours = image.copy()
    draw_contours_in_image(filtered_contours_1, image_with_filtered_contours)
    draw_contours_in_image(filtered_contours_2, image_with_filtered_contours)
    display_image_in_terminal(image_with_filtered_contours, "Perspective transformed image with filtered contours")

    # Approximate the contours in 'filtered_contours_1' and 'filtered_contours_2'
    approximated_contours_1 = approximate_contours(filtered_contours_1)
    approximated_contours_2 = approximate_contours(filtered_contours_2)

    # Get the coordinates of the corners of all contours from 'approximated_contours_1' and 'approximated_contours_2'
    corners_of_approximated_contours_1 = corner_detector(approximated_contours_1, pre_processed_image, 4)
    corners_of_approximated_contours_2 = corner_detector(approximated_contours_2, pre_processed_image, 4)
    image_with_corners = image.copy()
    draw_circles(corners_of_approximated_contours_1, image_with_corners)
    draw_circles(corners_of_approximated_contours_2, image_with_corners)
    display_image_in_terminal(image_with_corners, "Perspective transformed image with corners of the filtered contours")