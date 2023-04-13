import cv2
import os

# Load hat images
hat_images = []
hat_folder = "hats"
for filename in os.listdir(hat_folder):
    if filename.endswith(".png"):
        img = cv2.imread(os.path.join(hat_folder, filename), -1)
        hat_images.append(img)

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture from default camera
cap = cv2.VideoCapture(0)

# Set window properties
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', 800, 600)

# Define empty hat and hat selected index
current_hat = None
hat_index = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Determine hat position and size
        hat_height = int(h * 0.6)
        hat_width = int(w * 1.2)
        hat_x = int(x - (hat_width - w) / 2)
        hat_y = int(y - hat_height * 0.8)

        # Check if current hat exists and is the correct size
        if current_hat is not None and current_hat.shape[0] == hat_height and current_hat.shape[1] == hat_width:
            hat_img_resized = current_hat
        else:
            # Select new hat image
            current_hat = hat_images[hat_index % len(hat_images)]
            hat_img_resized = cv2.resize(current_hat, (hat_width, hat_height))

        # Create mask for hat
        alpha_h = hat_img_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_h

        # Add hat to the frame
        for c in range(0, 3):
            frame[hat_y:hat_y+hat_height, hat_x:hat_x+hat_width, c] = (alpha_h * hat_img_resized[:, :, c] + alpha_l * frame[hat_y:hat_y+hat_height, hat_x:hat_x+hat_width, c])

        # Increment hat index when R is pressed
        if cv2.waitKey(1) == ord('r'):
            hat_index += 1
            current_hat = None

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Exit program when 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
