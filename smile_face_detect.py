import cv2
import time
import os

# Folder to save images
output_dir = "smile_captures2"

# Create the folder if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Start webcam
cap = cv2.VideoCapture(0)

saved_count = 0
last_capture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.5, 20)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
            roi_smile = roi_color[sy:sy+sh, sx:sx+sw]
            # Save image only once every 2 seconds
            if time.time() - last_capture_time > 2:
                filename = f"{output_dir}/smile_{saved_count}.jpg"
                cv2.imwrite(filename, roi_smile)#roi_gray for full face,colour,frame not use roi
                print(f"[✔] Saved: {filename}")
                saved_count += 1
                last_capture_time = time.time()
            break

    cv2.imshow("Auto Smile Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
