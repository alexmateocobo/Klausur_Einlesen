import cv2
from utils_text_detector import approximate_contours
from utils_text_detector import corner_detector
from utils_text_detector import display_image_in_terminal
from utils_text_detector import draw_circles
from utils_text_detector import draw_contours_in_image
from utils_text_detector import extract_text_from_image
from utils_text_detector import generate_sqaure_grid
from utils_text_detector import hierarchical_contour_filtering_by_size
from utils_text_detector import pre_process_scanned_image

def text_detector(image):
    """
    Detects and extracts text from specific regions within an image assumed to contain a table.

    Steps:
    - Pre-process the image to enhance features for text detection.
    - Find contours within the pre-processed image.
    - Identify the largest contour as the external borders of the document.
    - Filter contours to isolate upper and lower rectangles assumed to represent table sections.
    - Approximate these contours to simplify corner detection.
    - Detect corners within the approximated contours for table corners visualization.
    - Generate a grid based on the detected corners.
    - Extract text from the image using the generated grid coordinates.

    Args:
    - image: The input image containing the table-like structure with text.

    Returns:
    - cropped_images_1: Extracted cropped images containing text from the first region.
    - cropped_images_2: Extracted cropped images containing text from the second region.
    """

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

    # Generate grid based on the points in 'corners_of_approximated_contours_1' and 'corners_of_approximated_contours_2'
    grid_1 = generate_sqaure_grid(corners_of_approximated_contours_1)
    grid_2 = generate_sqaure_grid(corners_of_approximated_contours_2)
    image_with_grids = image.copy()
    draw_circles(grid_1, image_with_grids)
    draw_circles(grid_2, image_with_grids)
    display_image_in_terminal(image_with_grids, "Perspective transformed image with corners of contours")

    # Extract cropped images from the pre-processed image using grid_1 and grid_2 coordinates
    cropped_images_1 = extract_text_from_image(pre_processed_image, grid_1)
    cropped_images_2 = extract_text_from_image(pre_processed_image, grid_2)
    display_image_in_terminal(cropped_images_1[0], "")
    display_image_in_terminal(cropped_images_1[1], "")
    display_image_in_terminal(cropped_images_1[2], "")
    display_image_in_terminal(cropped_images_1[3], "")

    return cropped_images_1, cropped_images_2