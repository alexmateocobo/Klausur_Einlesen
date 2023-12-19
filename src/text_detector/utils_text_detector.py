import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def approximate_contours(contours):
    """
    Approximates contours using the Douglas-Peucker algorithm.

    Args:
    - contours: List of input contours to be approximated.

    Returns:
    - approximated_contours: List containing approximated contours.
    """

    # Create an empty list to store the approximated contours
    approximated_contours = []

    # Set the error rate for approximation
    error_rate = 0.01

    for contour in contours:
        # Calculate the actual arc length of the input contour
        actual_arc_length = cv2.arcLength(contour, True)

        # Calculate the epsilon value for approximation based on the error rate and the arc length
        epsilon = error_rate * actual_arc_length

        # Approximate the input contour using the Douglas-Peucker algorithm
        approximated_contour = cv2.approxPolyDP(contour, epsilon, True)
        approximated_contours.append(approximated_contour)
    
    return approximated_contours

def corner_detector(contours, gray_image, max_corners_per_contour):
    """
    Applies Shi-Tomasi Corner Detector within specified contours to find corners.

    Args:
    - contours: List of contours within which corners are to be detected.
    - gray_image: Grayscale image on which the contours are detected.
    - max_corners_per_contour: Maximum number of corners per contour.

    Returns:
    - storage_list: List containing detected corners within the specified contours.
    """

    # Create an empty list to store the corners
    storage_list = []

    # Iterate through the list of contours
    for contour in contours:
        # Create a blank mask with the same dimensions as the image
        mask = np.zeros_like(gray_image)

        # Draw the contour on the mask
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        
        # Use Shi-Tomasi Corner Detector within the contour to find corners
        corners = cv2.goodFeaturesToTrack(mask, max_corners_per_contour, qualityLevel=0.001, minDistance=10)

        # Iterate through the list of corners
        for corner in corners:
            # Convert corner to tuple (x, y) and check if it's in the storage list
            corner = tuple(corner[0])
            if not point_is_in_storage_list(corner, storage_list):
                storage_list.append(corner)

    return storage_list

def display_image_in_terminal(image, title):
    """
    Displays an image in the terminal with title.

    Args:
    - image: The image to be displayed.
    - title: The title for the displayed image.
    """

    # Display the image using Matplotlib
    plt.figure(figsize = [10, 10])
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    plt.axis("off")
    plt.title(title)
    plt.show()

def draw_circles(points, image):
    """
    Draws circles on an image at specified points.

    Args:
    - points: 2D-array of (x, y) coordinate pairs.
    - image: The image on which circles will be drawn.
    """

    # Iterate through the points
    for row in points:
        for point in row:
            x, y = int(point[0]), int(point[1])  # Extract x, y coordinates as integers
            cv2.circle(image, (x, y), 10, (0, 255, 0), -1)  # Draw a filled green circle

def draw_contours_in_image(contours, image):
    """
    Draws contours on an image.

    Args:
    - contours: List of contours to be drawn.
    - image: The image on which contours will be drawn.
    """

    # Draw all contours on the image in green color with a thickness of 10
    cv2.drawContours(image, contours, -1, (0, 255, 0), 10)

def extract_text_from_image(image, array):
    """
    Extracts cropped images from an image using coordinates from the array.

    Args:
    - image: The original image (numpy.ndarray format).
    - array: Array containing coordinates for cropping regions.

    Returns:
    - List of cropped images extracted from the original image.
    """

    cropped_images = []
    max_i = len(array) - 1
    max_j = len(array[0]) - 1

    # Iterate through array indices for cropping
    for i in range(max_i):
        for j in range(max_j):
            if i + 1 < len(array) and j + 1 < len(array[0]):
                point_1 = array[i][j]
                point_4 = array[i + 1][j + 1]

                # Convert the image (numpy.ndarray) back to a PIL Image
                pil_image = Image.fromarray(image)

                # Use the PIL Image to crop the region
                cropped_image = pil_image.crop((point_1[0], point_1[1], point_4[0], point_4[1]))

                # Convert the PIL Image to a NumPy array
                cropped_image_np = np.array(cropped_image)
                cropped_images.append(cropped_image_np)

    return cropped_images

def find_unique_x_coordinates(points):
    """
    Finds unique x-coordinates from a list of points and returns them in ascending order.

    Args:
    - points: List of (x, y) coordinate pairs.

    Returns:
    - List containing unique x-coordinates from the given list of points, sorted in ascending order.
    """

    # Create an empty list to store all the unique x-coordinates
    storage_list = []

    # Loop through each point in the input list
    for point in points:
        x, y = point  # Extract x-coordinate from the point
        
        # Check if the x-coordinate is not already in the storage_list
        if not value_is_in_storage_list(x, storage_list):
            storage_list.append(x)  # Add the unique x-coordinate to the storage_list
    
    # Sort the storage_list in ascending order
    storage_list.sort()
    
    return storage_list

def find_unique_y_coordinates(points):
    """
    Finds unique y-coordinates from a list of points.

    Args:
    - points: List of (x, y) coordinate pairs.

    Returns:
    - List containing unique y-coordinates from the given list of points.
    """

    # Create an empty list to store all the unique y-coordinates
    storage_list = []

    # Loop through each point in the input list
    for point in points:
        x, y = point  # Extract y-coordinate from the point
        
        # Check if the y-coordinate is not already in the storage_list
        if not value_is_in_storage_list(y, storage_list):
            storage_list.append(y)  # Add the unique y-coordinate to the storage_list
    
    # Sort the storage_list in ascending order
    storage_list.sort()
    
    return storage_list

def generate_sqaure_grid(points):
    """
    Generates artificial corners for an image based on input points.

    Args:
    - points: List of points in the image.
    - image: The image for which corners are being generated.

    Returns:
    - List of artificial corners generated from the unique x and y coordinates of input points.
    """

    # Find all the unique x-coordinates from the input points
    unique_x_coordinates = find_unique_x_coordinates(points)

    # Find all the unique y-coordinates from the input points
    unique_y_coordinates = find_unique_y_coordinates(points)

    # Generate a grid of points based on unique x and y coordinates
    generated_grid_points = generate_grid_from_points(unique_x_coordinates, unique_y_coordinates)

    return generated_grid_points

def generate_grid_from_points(x_coordinates, y_coordinates):
    """
    Generates a grid of points from given x- and y-coordinates.

    Args:
    - x_coordinates: List of x-coordinates.
    - y_coordinates: List of y-coordinates.

    Returns:
    - 2D array representing the grid points, with nrow = len(y_coordinates) and ncol = len(x_coordinates).
    """
    # Initialize the 2D array with zeros
    grid = [[0 for _ in range(len(x_coordinates))] for _ in range(len(y_coordinates))]

    # Populate the grid with coordinates
    for i, x_coordinate in enumerate(x_coordinates):
        for j, y_coordinate in enumerate(y_coordinates):
            grid[j][i] = (x_coordinate, y_coordinate)

    return grid

def hierarchical_contour_filtering_by_size(contours, hierarchy, n):
    """
    Filters contours based on size hierarchy.

    Args:
    - contours: List of contours detected in the image.
    - hierarchy: Contour hierarchy information.
    - n: The rank of the contour by size to be used as a reference.

    Returns:
    - storage_list: Contours that are children of the nth largest contour.
    """

    # Create an empty list to store the contours that are children of the nth largest contour
    storage_list = []

    # Calculate areas of each contour
    areas = [cv2.contourArea(c) for c in contours]

    # Find the index of the nth largest contour based on area
    nth_largest_contour_index = areas.index(sorted(areas)[-n])

    # Iterate through the indices of contours
    for i in range(len(contours)):
        # Check if the contour is a child of the nth largest contour
        if hierarchy[0][i][3] == nth_largest_contour_index:
            # Append the contour to the list of filtered contours
            storage_list.append(contours[i])

    return storage_list

def point_is_in_storage_list(point, storage_list, tolerance=10):
    """
    Checks if a point is approximately in a storage list within a specified tolerance.

    Args:
    - point: The point to be checked.
    - storage_list: List of points to compare against.
    - tolerance: Tolerance distance to consider a point as 'approximately' in the list.

    Returns:
    - Boolean indicating whether a similar point was found within the tolerance.
    """

    for point_in_storage_list in storage_list:
        # Calculate the Euclidean distance between 'point' and 'point_in_storage_list'
        distance = np.linalg.norm(np.array(point) - np.array(point_in_storage_list))
        
        if distance < tolerance:
            return True  # Found a matching point within tolerance

    return False  # No matching point was found in the list

def pre_process_scanned_image(scanned_image):
    """
    Pre-processes an image of a scanned document.

    Args:
    - image: The original input image (in BGR format) to be pre-processed.

    Returns:
    - thresholded_image: The pre-processed image after converting to grayscale and applying
                          adaptive thresholding to enhance document features for analysis.
    """

    # Convert the original image to grayscale
    gray_image = cv2.cvtColor(scanned_image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image using adaptive thresholding
    thresholded_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 51, 15)

    return thresholded_image

def value_is_in_storage_list(value, storage_list, tolerance=10):
    """
    Checks if a value is approximately in a storage list within a specified tolerance.

    Args:
    - value: The value to be checked.
    - storage_list: List of values to compare against.
    - tolerance: Tolerance difference to consider a value as 'approximately' in the list.

    Returns:
    - Boolean indicating whether a similar value was found within the tolerance.
    """

    for value_in_storage_list in storage_list:
        if abs(value - value_in_storage_list) < tolerance:
            return True  # Found a matching value within tolerance

    return False  # No matching value was found in the list
