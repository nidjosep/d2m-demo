import cv2
import numpy as np
import subprocess
import os

from pose_estimator import PoseEstimator

class VideoProcessor:
    def __init__(self):
        """
        Initialize the VideoProcessor with a PoseEstimator instance to process video frames.
        """
        self.pose_estimator = PoseEstimator()

    def process_video(self, video_path):
        """
        Process a video file to estimate poses on each frame and compile the processed frames into a NumPy array.
        """
        # Initialize a list to store processed frames
        processed_frames = []
        try:
            # Attempt to open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Failed to open video file: {video_path}")

            # Read and process each frame of the video
            while True:
                success, frame = cap.read()
                if not success:
                    break  # End of video or failure to read frame

                # Process the current frame to extract pose and convert it into a sketchman frame
                processed_frame = self.process_frame(frame)
                processed_frames.append(processed_frame)

            # Release the video capture object
            cap.release()

            # Convert the list of processed frames into a NumPy array and return
            return np.array(processed_frames)
        except Exception as e:
            print(f"Error processing video: {e}")
            return None

    def process_frame(self, frame):
        """
        Process a single video frame to estimate the pose and convert it into a sketchman representation.
        """
        # Use the pose estimator to draw the pose on the frame
        sketchman_frame = self.pose_estimator.draw_sketchman_on_frame(frame)
        return sketchman_frame

    def frames_to_video(self, frames, audio_path, output_video_path, fps=30):
        """
        Combine processed frames into a video, add the original audio, and save the output.
        """
        # Concatenate frames along the first axis if they're split
        frames = np.concatenate(frames, axis=0)

        # Convert frames to BGR color format for video writing
        frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in frames]

        # Setup video writer object with MP4 codec, frame size, and fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].shape[:2]
        out = cv2.VideoWriter('temp_video.mp4', fourcc, fps, (width, height))

        # Write each frame to the temporary video file
        for frame in frames:
            out.write(frame)
        out.release()

        # Use ffmpeg to combine the video with audio and save to the specified output path
        subprocess.call(['ffmpeg', '-i', 'temp_video.mp4', '-i', audio_path,
                         '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental', output_video_path])
        subprocess.call(['ffmpeg', '-i', 'temp_video.mp4', '-i', audio_path,
                 '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental', output_video_path])

        os.remove('temp_video.mp4')  # Cleanup the temporary video file after merging
