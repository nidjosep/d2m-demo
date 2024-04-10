import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        # Initialize the Pose model with specific parameters.
        # 'min_detection_confidence=0.5' sets the minimum confidence value
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.4)

    def draw_sketchman_on_frame(self, frame):
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Create a blank canvas with the same dimensions as the input frame.
        canvas = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Check if any pose landmarks were detected in the frame.
        if results.pose_landmarks:
            # Draw each connection line between pose landmarks.
            for connection in self.mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection  # Indices of the start and end points.
                color = (255, 255, 255)  # White color for the sketch lines.

                # Retrieve the normalized landmark positions and convert them to pixel coordinates.
                start_landmark = results.pose_landmarks.landmark[start_idx]
                end_landmark = results.pose_landmarks.landmark[end_idx]

                # Convert normalized coordinates to pixel values for drawing.
                start_px = self.mp_drawing._normalized_to_pixel_coordinates(start_landmark.x, start_landmark.y, frame.shape[1], frame.shape[0])
                end_px = self.mp_drawing._normalized_to_pixel_coordinates(end_landmark.x, end_landmark.y, frame.shape[1], frame.shape[0])

                # Draw a line between the start and end points if both are detected.
                if start_px and end_px:
                    cv2.line(canvas, start_px, end_px, color, thickness=8, lineType=cv2.LINE_AA)

            # Draw circles for each detected landmark.
            for landmark in results.pose_landmarks.landmark:
                landmark_px = self.mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, frame.shape[1], frame.shape[0])
                if landmark_px:
                    cv2.circle(canvas, landmark_px, 10, color, -1, lineType=cv2.LINE_AA)

        # Resize the canvas to the target size (96x96 pixels) after drawing.
        scaled_canvas = cv2.resize(canvas, (96, 96))
        return scaled_canvas
