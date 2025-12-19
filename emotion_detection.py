from fer import FER
import cv2

# Initialize FER
detector = FER()

# Open the video capture (change the index if needed)
cam = cv2.VideoCapture(0)  

while True:
    success, frame = cam.read()
    if success:
        # Detect emotions in the frame
        emotions = detector.detect_emotions(frame)
        
        # Debug: Print out the detected emotions
        print("Detected emotions:", emotions)
        
        # Process each detected face
        for face in emotions:
            # Extract emotion data
            emotion_data = face['emotions']
            
            # Debug: Print out the emotion data
            print("Emotion data:", emotion_data)
            
            if emotion_data:
                # Find the emotion with the highest probability
                max_emotion = max(emotion_data, key=emotion_data.get)
                max_prob = emotion_data[max_emotion]
                
                # Get bounding box coordinates (if available) for positioning the text
                (x, y, w, h) = face['box']
                
                # Display the emotion and its probability on the frame
                cv2.putText(frame, f'{max_emotion}: {max_prob*100:.2f}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display the frame with recognized emotions
        cv2.imshow("Frame", frame)
        
    # Check for the 'Esc' key to break the loop
    key = cv2.waitKey(1)
    if key == 27:  # 27 is the ASCII code for 'Esc' key
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
