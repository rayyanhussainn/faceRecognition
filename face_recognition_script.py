import cv2
import face_recognition

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load a known image and encode the face
known_image = face_recognition.load_image_file("heidi.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Store known face encodings and corresponding names
known_faces = [known_face_encoding]
known_names = ["Person Name"]

# Start webcam feed
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame to grayscale (for face detection)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Convert the frame to RGB (for face recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find face encodings in the current frame
    current_face_encodings = face_recognition.face_encodings(rgb_frame)

    # Loop over each face detected by OpenCV
    for (x, y, w, h), face_encoding in zip(faces, current_face_encodings):
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Compare the face encoding to known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)

        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        # Display the name below the rectangle
        cv2.putText(frame, name, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Press 'q' to quit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()
