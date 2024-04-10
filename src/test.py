# Import necessary libraries
from audio_processor import AudioProcessor
from video_processor import VideoProcessor
from utils import Utils
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import librosa
import soundfile as sf
import subprocess
import gc
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import cv2
from tensorflow.keras.models import load_model



def main():
    base_path = '.' #f'/content/drive/MyDrive/d2m'
    raw_input_path = f'{base_path}/input/dataset/raw'

    # Initializes VideoProcessor and AudioProcessor for processing raw dataset
    video_processor = VideoProcessor()
    audio_processor = AudioProcessor()

    vae = load_model('m.snapshots/snapshot-midi/vae.h5')
    vae.summary()
    output_path = Utils.create_timestamped_output_folder(f'{base_path}/output')
    video_data = Utils.load_data(f'{base_path}/input/dataset/raw', video_processor, audio_processor)


    # Initialize the video capture object
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    processed_frames = []  # Initialize the frame queue

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Add the frame to our queue
            processed_frames.append(video_processor.process_frame(frame))

            # Check if we have collected 32 frames
            if len(processed_frames) == 32:
                video_data = []
                video_frames_ = Utils.normalize_video(processed_frames)
                audio_features_ = np.random.rand(32, 128, 1)
                video_data.append((audio_features_, video_frames_))
                audio_test, video_test = Utils.prepare_data(video_data)

                #audio_test, _, _ = Utils.normalize_audio(audio_test)
                video_test = Utils.normalize_video(video_test)

                x__, _ = vae.predict([audio_test, video_test])
                x__ = audio_processor.features_to_audio(x__)
                print("Converted ")

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # When everything done, release the capture and destroy all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()



main()
