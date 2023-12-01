import cv2
import matplotlib as plt
import numpy as np

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

def crop_image_percentage_based(image, percentage):
    """
    Crops an image based on a specified percentage of its size.

    Args:
    - image: The input image to be cropped.
    - percentage: The percentage of the image size to be cropped.

    Returns:
    - cropped_image: The cropped portion of the input image.
    """

    # Calculate percentage of image size for cropping
    height, width = image.shape[:2]
    crop_percent = percentage / 100

    # Calculate the cropping dimensions
    top_crop = int(height * crop_percent)
    bottom_crop = int(height * crop_percent)
    left_crop = int(width * crop_percent)
    right_crop = int(width * crop_percent)

    # Apply the cropping
    cropped_image = image[top_crop:-bottom_crop, left_crop:-right_crop]

    return cropped_image

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

def draw_contours_in_image(contours, image):
    """
    Draws contours on an image.

    Args:
    - contours: List of contours to be drawn.
    - image: The image on which contours will be drawn.
    """

    # Draw all contours on the image in green color with a thickness of 15
    cv2.drawContours(image, contours, -1, (0, 255, 0), 15)

def draw_labeled_circles(points, image):
    """
    Draws labeled circles on an image at specified points.

    Args:
    - points: List of (x, y) coordinate pairs.
    - image: The image on which circles and labels will be drawn.
    """

    # Iterate through the points
    for i, point in enumerate(points):
        x, y = int(point[0]), int(point[1])  # Extract x, y coordinates as integers
        cv2.circle(image, (x, y), 20, (0, 255, 0), -1)  # Draw a filled green circle

        # Add text label with point index
        cv2.putText(image, f'{i}', (x - 10, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

def final_destination_corners(corners):
    """
    Calculates the final destination corners for perspective correction.

    Args:
    - corners: Tuple of corners obtained from the perspective transformation.

    Returns:
    - destination_corners: Numpy array containing the final destination coordinates.
    """

    # Unpack the corners tuple
    (top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner) = corners

    # Calculate the maximum width based on the horizontal distance between the corners
    widthA = np.sqrt(((bottom_right_corner[0] - bottom_left_corner[0]) ** 2) + ((bottom_right_corner[1] - bottom_left_corner[1]) ** 2))
    widthB = np.sqrt(((top_right_corner[0] - top_left_corner[0]) ** 2) + ((top_right_corner[1] - top_left_corner[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Calculate the maximum height based on the vertical distance between the corners
    heightA = np.sqrt(((top_right_corner[0] - bottom_right_corner[0]) ** 2) + ((top_right_corner[1] - bottom_right_corner[1]) ** 2))
    heightB = np.sqrt(((top_left_corner[0] - bottom_left_corner[0]) ** 2) + ((top_left_corner[1] - bottom_left_corner[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Define the final destination coordinates as a numpy array
    destination_corners = np.array([[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]])

    return destination_corners

def order_points_clockwise(points):
    """
    Rearranges coordinates to order: top-left, top-right, bottom-right, bottom-left.

    Args:
    - points: List of coordinates to be rearranged.

    Returns:
    - rectangle_corners: List containing reordered coordinates.
    """

    # Initialize an array to hold the rearranged points
    rectangle_corners = np.zeros((4, 2), dtype='float32')

    # Convert points to a NumPy array
    points = np.array(points)

    # Calculate the sum of x and y coordinates for each point
    s = points.sum(axis=1)

    # Top-left point will have the smallest sum.
    rectangle_corners[0] = points[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rectangle_corners[2] = points[np.argmax(s)]

    # Calculate the differences between x and y coordinates for each point
    differences_of_coordinates = np.diff(points, axis=1)

    # Top-right point will have the smallest difference.
    rectangle_corners[1] = points[np.argmin(differences_of_coordinates)]
    # Bottom-left will have the largest difference.
    rectangle_corners[3] = points[np.argmax(differences_of_coordinates)]

    # Convert and return the coordinates to integers and then to a list
    return rectangle_corners.astype('int').tolist()

def perspective_transformation(source_corners, destination_corners, image):
    """
    Applies perspective transformation to an image using source and destination corners.

    Args:
    - source_corners: Source corners obtained from the image.
    - destination_corners: Final destination corners for perspective correction.
    - image: The image on which perspective transformation will be applied.

    Returns:
    - perspective_transformed_image: Image after applying perspective transformation.
    """

    # Get the homography matrix using source and destination corners
    M = cv2.getPerspectiveTransform(np.float32(source_corners), np.float32(destination_corners))

    # Perspective transformation using the homography matrix
    perspective_transformed_image = cv2.warpPerspective(image, M, (destination_corners[2][0], destination_corners[2][1]), flags = cv2.INTER_LINEAR)

    return perspective_transformed_image

def point_is_in_storage_list(point, storage_list, tolerance = 15):
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

def pre_process_image(image):
    """
    Pre-processes an image of a document.

    Args:
    - image: The original input image (in BGR format) to be pre-processed.

    Returns:
    - eroded_image: The pre-processed image after performing grayscale conversion,
                    blurring, morphology operations (closing), edge detection, dilation,
                    and erosion to prepare it for document analysis.
    """

    # Convert the original image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_image_in_terminal(gray_image, "Grayscale image")

    # Blurr the grayscale image
    blurred_image = cv2.GaussianBlur(gray_image, (15, 15), 0)
    display_image_in_terminal(blurred_image, "Blurred image")

    # Apply dilation and erosion to remove text from the blurred image
    kernel = np.ones((15, 15), np.uint8)
    closed_image = cv2.morphologyEx(blurred_image, cv2.MORPH_CLOSE, kernel, iterations=3)
    display_image_in_terminal(closed_image, "Closed image")

    # Edge 'closed_image'
    edged_image = cv2.Canny(closed_image, 5, 150)
    display_image_in_terminal(edged_image, "Edged image")

    # Create a 5x5 kernel filled with ones
    kernel = np.ones((5, 5))

    # Dilate the edged image
    dilated_image = cv2.dilate(edged_image, kernel, iterations = 2)
    display_image_in_terminal(dilated_image, "Dilated image")

    # Erode the dilated image
    eroded_image = cv2.erode(dilated_image, kernel, iterations = 1)
    display_image_in_terminal(eroded_image, "Eroded image")

    return eroded_image

