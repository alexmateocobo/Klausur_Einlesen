import cv2
import numpy as np
from utils_document_scanner import corner_detector
from utils_document_scanner import crop_image_percentage_based
from utils_document_scanner import display_image_in_terminal
from utils_document_scanner import draw_contours_in_image
from utils_document_scanner import draw_labeled_circles
from utils_document_scanner import final_destination_corners
from utils_document_scanner import order_points_clockwise
from utils_document_scanner import perspective_transformation
from utils_document_scanner import pre_process_image

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

def document_scanner(image_name):
    # Read the original image
    image = cv2.imread(image_name)
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

    return cropped_image