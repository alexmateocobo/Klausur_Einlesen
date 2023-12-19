import cv2
import time
#Main Datei!!!
# initialize the webcam and pass a constant which is 0
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

HIGH_VALUE = 10000
WIDTH = HIGH_VALUE
HEIGHT = HIGH_VALUE

focus = 35

fourcc = cv2.VideoWriter_fourcc(*'XVID')
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

time.sleep(1)
print(width,height)

# cam.set(cv2.CAP_PROP_FPS, 30.0)
# cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
# cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# title of the app
cv2.namedWindow('python webcam screenshot app')

# let's assume the number of images gotten is 0
img_counter = 0

latest_img = None  # To store the latest image

# Check if the camera is opened
if not cam.isOpened():
    print("Cannot open camera")
    exit()

# Set the focus property IIIII MUSS VERBESSERT WERDEN IIIIII
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
cam.set(cv2.CAP_PROP_FOCUS, focus)  # Adjust focus value (this value might need adjustments)

# while loop
while True:
    # initializing the frame, ret
    ret, frame = cam.read()
    
    # if statement
    if not ret:
        print('failed to grab frame')
        break
    # the frame will show with the title of test
    cv2.imshow('test', frame)
    # to get continuous live video feed from my laptop's webcam
    k = cv2.waitKey(1)
    # if the escape key is pressed, the app will stop
    if k % 256 == 27:
        print('Escape hit, closing the app')
        break
    # if the spacebar key is pressed, screenshots will be taken
    elif k % 256 == 32:
        
        cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        cam.set(cv2.CAP_PROP_FOCUS, focus)

        time.sleep(3)
        
        # the format for storing the images screenshotted
        img_name = f'open__{focus}_frame_{img_counter}.jpg'
        # saves the image as a jpg file
        cv2.imwrite(img_name, frame)
        print('Screenshot taken')
        # the number of images automatically increases by 1
        img_counter += 1
        latest_img = cv2.imread(img_name)  # Load the latest image after it's saved

        #focus += 5


        # Show the latest image if available
        if latest_img is not None:
            cv2.imshow('Latest Image Number:' + str(img_counter-1), latest_img)

# release the camera
cam.release()

# stops the camera window
cv2.destroyAllWindows()
