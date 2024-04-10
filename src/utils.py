import matplotlib.pyplot as plt
import datetime
import librosa
import numpy as np
import os
import base64
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.manifold import TSNE

class Utils:

    @staticmethod
    def plot_training_history(training_history):
        """
        Plots traning history
        """
        # Plot training history
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(training_history.history['loss'], label='Training Loss')
        plt.plot(training_history.history['val_loss'], label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

    def normalize_audio(data):
        """
        Normalize audio data for model processing.
        """
        data = np.repeat(data[..., np.newaxis], 1, -1)
        data = data / 127
        return data

    @staticmethod
    def normalize_video(data):
        """
        Normalize video frame data for model processing.
        """
        # Adjusts channel dimension for model compatibility
        data = np.repeat(data[..., np.newaxis], 1, -1)
        # Normalize image frames to a range between 0 and 1
        data = data / 255.0
        return data

    @staticmethod
    def denormalize_audio(data):
        """
        Denormalizes audio data to its original value range.
        """
        # Select the first channel for grayscale data
        data = data[..., 0]
        # Scale data back to its original range
        data = data * 127
        return data

    @staticmethod
    def denormalize_video(data):
        """
        Denormalizes video data to its original value range.
        """
        # Select the first channel for grayscale data
        data = data[..., 0:1]
        # Scale data back to the range 0-255 and convert to uint8
        data = (data * 255).astype(np.uint8)
        return data

    @staticmethod
    def create_sequences(audio_features, video_frames, sequence_length=32):
        """
        Creates sequences of audio features and video frames for temporal processing.
        """
        sequences = []
        # Generate sequences of specified length
        for i in range(0, min(len(audio_features), len(video_frames)) - sequence_length + 1, sequence_length):
            audio_seq = audio_features[i:i+sequence_length]
            video_seq = video_frames[i:i+sequence_length]
            sequences.append((audio_seq, video_seq))
        return sequences

    @staticmethod
    def show_sample(frame):
        """
        Displays a single video frame.
        """
        plt.imshow(frame, cmap='gray')
        plt.title('Sample Frame')
        plt.axis('off')
        plt.show()

    @staticmethod
    def display_spectrogram(waveform, sr=22050, title='Spectrogram', figsize=(10, 4)):
        """
        Displays a spectrogram of an audio waveform.
        """
        # Convert waveform to a dB-scaled spectrogram
        spectogram = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
        plt.figure(figsize=figsize)
        # Display spectrogram
        librosa.display.specshow(spectogram, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.show()

    @staticmethod
    def create_timestamped_output_folder(base_output_path):
        """
        Creates a new output folder with a timestamp.
        """
        # Create a timestamp string
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Construct the new folder path
        new_folder_path = os.path.join(base_output_path, timestamp)
        # Create the folder
        os.makedirs(new_folder_path, exist_ok=True)
        return new_folder_path

    @staticmethod
    def is_defined_and_nonempty(variable_name):
        """
        Checks if a variable is defined in the global scope and is not empty.
        """
        return variable_name in globals() and len(globals()[variable_name])

    @staticmethod
    def load_data(raw_input_path, video_processor, audio_processor, isTraining = True):
        """
        Loads and processes video and audio data from a specified directory.
        """
        # List all .mp4 files in the directory
        video_files = [f for f in os.listdir(raw_input_path) if f.endswith('.mp4')]
        video_files = sorted(video_files)  # Sort the files for consistent order

        video_data = []  # Initialize a list to store processed video and audio data

        # Process each video file in the directory
        for video_file in video_files:
            path = os.path.join(raw_input_path, video_file)  # Get the full path of the video file
            print(f"Processing: {path}")

            # Process video to extract frames
            video_frames_ = video_processor.process_video(path)
            if isTraining:
                # Process audio to extract features
                audio_features_ = audio_processor.extract_feature(path)
            else:
                audio_features_ = np.zeros((video_frames_.shape[0], 128))

            # Append the processed audio and video data as a tuple to the video_data list
            video_data.append((audio_features_, video_frames_))

        return video_data

    @staticmethod
    def prepare_data(video_data):
        """
        Prepares video and audio data for training by creating sequences.
        """
        sequences = []
        for audio_features_, video_frames_ in video_data:
            # For each pair of audio features and video frames,
            # generate sequences that are split into smaller,
            # equal-length segments for training.
            sequences_ = Utils.create_sequences(audio_features_, video_frames_)
            # Extend the main list of sequences with the newly created segments.
            sequences.extend(sequences_)

        # Unzip the sequences into separate lists for audio and video
        audio_sequences, video_sequences = zip(*sequences)

        return np.array(audio_sequences), np.array(video_sequences)

    @staticmethod
    def visualize_piano_roll(piano_roll_matrix,fs=10,min_note=0,max_note=128) :
        #piano_roll_matrix = np.squeeze(piano_roll_matrix, axis=-1)
        piano_roll_matrix = piano_roll_matrix.T
        # Use librosa's specshow function for displaying the piano roll
        librosa.display.specshow(piano_roll_matrix[min_note:max_note],hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',fmin=pretty_midi.note_number_to_hz(min_note))
        plt.colorbar(format='%+2.0f dB')
        plt.title('MIDI')
        plt.show()
        return piano_roll_matrix

    def filter_below_percent_of_mean(piano_roll_matrix,threshold = 0.85):
      # Calculate the mean value along the first dimension
      mean_value = np.max(piano_roll_matrix, axis=0)

      # Calculate the threshold value (80% of the mean)
      threshold_value = threshold * mean_value

      # Find the indices where the value is below the threshold
      below_threshold_indices = np.where(piano_roll_matrix < threshold_value)

      # Set the values below the threshold to zero
      filtered_piano_roll_matrix = np.copy(piano_roll_matrix)
      filtered_piano_roll_matrix[below_threshold_indices] = 0

      return filtered_piano_roll_matrix


    def piano_roll_to_pretty_midi(piano_roll, fs=10, program=0):
        '''Convert a Piano Roll array into a PrettyMidi object
        with a single instrument.

        '''
        notes, frames = piano_roll.shape
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=program)

        # pad 1 column of zeros so we can acknowledge inital and ending events
        piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

        # use changes in velocities to find note on / note off events
        velocity_changes = np.nonzero(np.diff(piano_roll).T)

        # keep track on velocities and note on times
        prev_velocities = np.zeros(notes, dtype=int)
        note_on_time = np.zeros(notes)

        for time, note in zip(*velocity_changes):
            # use time + 1 because of padding above
            velocity = piano_roll[note, time + 1]
            time = time / fs
            if velocity > 0:
                if prev_velocities[note] == 0:
                    note_on_time[note] = time
                    prev_velocities[note] = velocity
            else:
                pm_note = pretty_midi.Note(
                    velocity=prev_velocities[note],
                    pitch=note,
                    start=note_on_time[note],
                    end=time)
                instrument.notes.append(pm_note)
                prev_velocities[note] = 0
        pm.instruments.append(instrument)
        return pm


    def convert_midi_to_audio(midi_file_path, output_file_path):
        fs = FluidSynth('fluids.sf2')
        fs.midi_to_audio(midi_file_path, output_file_path)

    def filter_below_percent_of_mean(piano_roll_matrix,threshold = 0.85):
        # Calculate the mean value along the first dimension
        mean_value = np.max(piano_roll_matrix, axis=0)

        # Calculate the threshold value (80% of the mean)
        threshold_value = threshold * mean_value

        # Find the indices where the value is below the threshold
        below_threshold_indices = np.where(piano_roll_matrix < threshold_value)

        # Set the values below the threshold to zero
        filtered_piano_roll_matrix = np.copy(piano_roll_matrix)
        filtered_piano_roll_matrix[below_threshold_indices] = 0

        return filtered_piano_roll_matrix


    def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
        '''Convert a Piano Roll array into a PrettyMidi object
        with a single instrument.

        '''
        notes, frames = piano_roll.shape
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=program)

        # pad 1 column of zeros so we can acknowledge inital and ending events
        piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

        # use changes in velocities to find note on / note off events
        velocity_changes = np.nonzero(np.diff(piano_roll).T)

        # keep track on velocities and note on times
        prev_velocities = np.zeros(notes, dtype=int)
        note_on_time = np.zeros(notes)

        for time, note in zip(*velocity_changes):
            # use time + 1 because of padding above
            velocity = piano_roll[note, time + 1]
            time = time / fs
            if velocity > 0:
                if prev_velocities[note] == 0:
                    note_on_time[note] = time
                    prev_velocities[note] = velocity
            else:
                pm_note = pretty_midi.Note(
                    velocity=prev_velocities[note],
                    pitch=note,
                    start=note_on_time[note],
                    end=time)
                instrument.notes.append(pm_note)
                prev_velocities[note] = 0
        pm.instruments.append(instrument)
        return pm