import cv2

cap = cv2.VideoCapture(0)
hat_img = cv2.imread('hat.png', -1)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection code:
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Hat placement code:
        hat_width = int(w * 1.5)
        hat_height = int(hat_width * hat_img.shape[0] / hat_img.shape[1])
        hat_x = x - int((hat_width - w) / 2)
        hat_y = y - hat_height + int(h / 5)

        if hat_x < 0:
            hat_x = 0
        if hat_x + hat_width > frame.shape[1]:
            hat_x = frame.shape[1] - hat_width
        if hat_y < 0:
            hat_y = 0
        if hat_y + hat_height > frame.shape[0]:
            hat_y = frame.shape[0] - hat_height

        hat_alpha = hat_img[:, :, 3] / 255.0
        hat_rgb = hat_img[:, :, :3]
        hat_resized = cv2.resize(hat_rgb, (hat_width, hat_height))
        hat_alpha_resized = cv2.resize(hat_alpha, (hat_width, hat_height))

        overlay = hat_resized.copy()
        background = frame[hat_y:hat_y + hat_height, hat_x:hat_x + hat_width]

        for c in range(3):
            overlay[:, :, c] = hat_resized[:, :, c] * hat_alpha_resized
            background[:, :, c] = background[:, :, c] * (1 - hat_alpha_resized)

        result = cv2.addWeighted(overlay, 1, background, 1, 0)
        frame[hat_y:hat_y + hat_height, hat_x:hat_x + hat_width] = result

    cv2.imshow('Hat on Head', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
