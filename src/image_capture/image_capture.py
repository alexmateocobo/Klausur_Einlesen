import cv2
import time

def capture_image():
    """
    Captures images from a webcam and allows users to take screenshots by pressing the spacebar.
    
    Steps:
    - Initialize the camera and set initial configurations.
    - Create a window to display the camera feed.
    - Continuously capture frames and display them in the window.
    - Press 'Spacebar' to take a screenshot, which is saved with a filename indicating the focus value.
    - Display the latest captured image in a separate window.
    
    Returns:
    - latest_img: The latest captured image.
    """

    # Initialize the camera
    cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    # Set initial camera settings
    HIGH_VALUE = 10000
    WIDTH = HIGH_VALUE
    HEIGHT = HIGH_VALUE
    focus = 35

    # Set video writer and frame properties
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pause to ensure camera setup
    time.sleep(1)
    print(width, height)

    # Create a window to display the camera feed
    cv2.namedWindow('python webcam screenshot app')

    # Initialize variables
    img_counter = 0
    latest_img = None

    # Check if the camera is opened successfully
    if not cam.isOpened():
        print("Cannot open camera")
        exit()

    # Set camera properties - focusing
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
    cam.set(cv2.CAP_PROP_FOCUS, focus)  # Set the focus value (may need adjustments)

    # Continuous loop for displaying and capturing frames
    while True:
        # Read frame and check if successful
        ret, frame = cam.read()
        
        # Check if frame retrieval failed
        if not ret:
            print('Failed to grab frame')
            break
        
        # Display the frame in a window titled 'test'
        cv2.imshow('test', frame)
        
        # Wait for a key press
        k = cv2.waitKey(1)
        
        # Check for key presses
        if k % 256 == 27:  # If ESC key is pressed, exit the app
            print('Escape hit, closing the app')
            break
        elif k % 256 == 32:  # If spacebar is pressed, take a screenshot
            cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
            cam.set(cv2.CAP_PROP_FOCUS, focus)

            time.sleep(3)
            
            # Define image file name and save the image as a jpg file
            img_name = f'open__{focus}_frame_{img_counter}.jpg'
            cv2.imwrite(img_name, frame)
            print('Screenshot taken')
            
            # Increment the image counter
            img_counter += 1
            # Load the latest image after it's saved
            latest_img = cv2.imread(img_name)  

            # Show the latest image if available
            if latest_img is not None:
                cv2.imshow('Latest Image Number:' + str(img_counter - 1), latest_img)

    # Release the camera and close the window
    cam.release()
    cv2.destroyAllWindows()

    return latest_img

