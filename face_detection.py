import cv2
import face_recognition
import numpy as np

# Define custom colors for each facial landmark group
FEATURE_COLORS = {
    'chin': (255, 0, 0),              # Blue
    'left_eyebrow': (0, 255, 0),      # Green
    'right_eyebrow': (0, 255, 255),   # Yellowish
    'nose_bridge': (255, 255, 0),     # Cyan
    'nose_tip': (255, 0, 255),        # Magenta
    'left_eye': (0, 0, 255),          # Red
    'right_eye': (255, 128, 0),       # Orange
    'top_lip': (128, 0, 128),         # Purple
    'bottom_lip': (0, 128, 128)       # Teal
}

def draw_face_landmarks(image, face_landmarks_list):
    """
    Draws colored circles on facial landmarks detected in the image.
    Each landmark group has a unique color.
    """
    for landmarks in face_landmarks_list:
        for feature, points in landmarks.items():
            color = FEATURE_COLORS.get(feature, (200, 200, 200))  # Fallback color
            for point in points:
                cv2.circle(image, point, 2, color, -1)  # Draw filled circle at each point

def add_legend_panel(frame):
    """
    Adds a vertical legend panel on the left side of the video frame
    to label facial features with corresponding colors.
    """
    legend_width = 200
    height = frame.shape[0]

    # Create a dark panel of the same height as the frame
    legend_panel = np.ones((height, legend_width, 3), dtype=np.uint8) * 30

    # Draw legend items (colored circles + text labels)
    y_offset = 30
    for feature, color in FEATURE_COLORS.items():
        cv2.circle(legend_panel, (20, y_offset - 5), 6, color, -1)  # Draw legend circle
        cv2.putText(legend_panel, feature, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 30

    # Horizontally stack the legend panel and the video frame
    combined = np.hstack((legend_panel, frame))
    return combined

def main():
    """
    Main function to capture video, detect faces and landmarks,
    draw visual overlays, and display output with a legend.
    """
    video_capture = cv2.VideoCapture(0)  # Start video capture from default camera

    if not video_capture.isOpened():
        print("Error: Cannot access camera.")
        return

    try:
        while True:
            # Read a frame from the camera
            success, frame = video_capture.read()
            if not success:
                print("Warning: Failed to capture frame.")
                continue

            # Mirror the frame for natural interaction
            frame = cv2.flip(frame, 1)

            # Convert BGR (OpenCV) to RGB (for face_recognition)
            rgb_frame = frame[:, :, ::-1]

            # Detect face locations and landmarks
            face_locations = face_recognition.face_locations(rgb_frame)
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

            for (top, right, bottom, left) in face_locations:
                # Expand the rectangle boundary around the face
                padding = 20
                h, w = frame.shape[:2]
                left = max(0, left - padding)
                top = max(0, top - padding)
                right = min(w, right + padding)
                bottom = min(h, bottom + padding)

                # Draw the green rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 150, 0), 2)

            # Draw the facial landmark points on the frame
            draw_face_landmarks(frame, face_landmarks_list)

            # Attach the legend panel on the left
            output_frame = add_legend_panel(frame)

            # Display the result in a window
            cv2.imshow('Face Detection with Landmarks & Legend', output_frame)

            # Exit the loop if any key is pressed
            if cv2.waitKey(1) > 0:
                break

    finally:
        # Release camera and close windows
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
