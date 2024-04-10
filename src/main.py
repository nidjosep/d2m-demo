# Import necessary libraries
from audio_processor import AudioProcessor
from video_processor import VideoProcessor
from utils import Utils
from vae_model import VAEModel
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

# For traning, the preprocessed dataset is only loaded on demand as batches
class MemoryMappedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, audio_filepath, video_filepath, indices, batch_size=16, shuffle=True):
        """
        Custom data generator for loading batches from memory-mapped NumPy arrays.

        Parameters:
        - audio_filepath: Path to the memory-mapped NumPy array file for audio data.
        - video_filepath: Path to the memory-mapped NumPy array file for video data.
        - indices: Array of indices defining which samples to use (enables k-fold functionality).
        - batch_size: Number of samples per batch.
        - shuffle: Whether to shuffle indices each epoch.
        """
        self.audio_data = np.load(audio_filepath, mmap_mode='r')
        self.video_data = np.load(video_filepath, mmap_mode='r')
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Compute number of batches to produce.
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch.
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data for the current batch.
        audio_batch = self.audio_data[batch_indices]
        video_batch = self.video_data[batch_indices]

        return [audio_batch, video_batch], [audio_batch, video_batch]  # Assuming an autoencoder structure

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is true.
        if self.shuffle:
            np.random.shuffle(self.indices)

class PredictionDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, audio_data, video_data, batch_size=32):
        self.audio_data = audio_data
        self.video_data = video_data
        self.batch_size = batch_size
        self.indices = np.arange(audio_data.shape[0])

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        audio_batch = self.audio_data[batch_indices]
        video_batch = self.video_data[batch_indices]
        return [audio_batch, video_batch]
    


def main():
    base_path = '.' #f'/content/drive/MyDrive/d2m'
    raw_input_path = f'{base_path}/input/dataset/raw'

    # Initializes VideoProcessor and AudioProcessor for processing raw dataset
    video_processor = VideoProcessor()
    audio_processor = AudioProcessor()

    output_path = Utils.create_timestamped_output_folder(f'{base_path}/output')
    video_data = Utils.load_data(f'{base_path}/input/dataset/raw', video_processor, audio_processor)

    # Align and prepare the data for training
    audio_train, video_train = Utils.prepare_data(video_data)

    # Normalize the training data
    audio_train, a_min, a_max = Utils.normalize_audio(audio_train)
    video_train = Utils.normalize_video(video_train)

    # For training, use the same dataset for validation to maximize the amount of data available for training.
    audio_val = audio_train
    video_val = video_train

    audio_shape = audio_train.shape   # Shape format: (timesteps, fequency bins, channels)
    video_shape = video_train.shape   # Shape format: (timesteps, height, width, channels)

    # Save the preprocessed datasets to disk for reuse
    np.save(f'{base_path}/input/dataset/preprocessed/audio_train.npy', audio_train)
    np.save(f'{base_path}/input/dataset/preprocessed/min_max_values.npy', np.array([a_min, a_max]))
    np.save(f'{base_path}/input/dataset/preprocessed/video_train.npy', video_train)

    audio_filepath = f'{base_path}/input/dataset/preprocessed/audio_train.npy'
    video_filepath = f'{base_path}/input/dataset/preprocessed/video_train.npy'

    # Load dataset if it has not been already loaded.
    if not Utils.is_defined_and_nonempty('audio_train'):
        audio_train = np.load(f'{audio_filepath}')
        min_max_values = np.load(f'{base_path}/input/dataset/preprocessed/min_max_values.npy')
        a_min, a_max = min_max_values[0], min_max_values[1]

    if not Utils.is_defined_and_nonempty('video_train'):
        video_train = np.load(f'{video_filepath}')

    print(f'audio_train.shape: {audio_train.shape}, video_train.shape: {video_train.shape}')

    # Denormalize the training data for displaying some samples
    audio_train_ = Utils.denormalize_audio(audio_train, a_min, a_max)
    video_train_ = Utils.denormalize_video(video_train)

    sample_audio_train_ = audio_train_[6:12]
    sample_video_train_ = video_train_[6:12]

    waveform_orig = audio_processor.features_to_audio(sample_audio_train_)

    print('\nAudio Spectrogram (2D)\n')
    Utils.display_spectrogram(waveform_orig, title='Original Spectrogram')

    # Display a few sample frames from the denormalized video data to verify the denormalization process.
    print('\nA sample sketchman Frames\n')
    Utils.show_sample(sample_video_train_[2][2])

    # Generate a sketchman video from pre-processed dataset to show how our sketchman based dataset
    # is different from the raw dance video dataset which is no more of significance
    print('\nA sample processed traning video sample (generated from raw video)\n')

    waveform_orig = audio_processor.features_to_audio(sample_audio_train_)
    intermediate = f'{output_path}/preprocessed_intermediate'
    audio_processor.save_audio(waveform_orig, f'{intermediate}.wav')
    video_processor.frames_to_video(sample_video_train_, f'{intermediate}.wav', f'{intermediate}.mp4')
    os.remove(f'{intermediate}.wav')
    Utils.render_video(f'{intermediate}.mp4')


    audio_shape = audio_train.shape[1:]   # (batch, timesteps, features)
    video_shape = video_train.shape[1:]   # (timesteps, height, width, channels)

    total_samples = audio_train.shape[0]  # Total number of samples in the dataset.

    # Initialize VAE model
    vae_model = VAEModel(audio_shape, video_shape)
    vae_model.vae.summary()

    weights_path = f'{base_path}/m.snapshots/vae_model_weights'

    # Load the latest checkpoint
    latest_checkpoint = tf.train.latest_checkpoint(f'{base_path}/m.snapshots')
    if latest_checkpoint:
        print("Loading weights from:", latest_checkpoint)
        vae_model.vae.load_weights(latest_checkpoint)
    else:
        print("No checkpoint found at:", f'{base_path}/m.snapshots')


    print("Training Started..")

    # Early Stopping handler to halt training when the loss stops improving.
    early_stopping_callback = EarlyStopping(
        monitor='loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )


    # Create a callback that saves the model's weights
    checkpoint_callback = ModelCheckpoint(
        filepath=weights_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    # Generate a range of indices that correspond to all samples
    total_indices = np.arange(total_samples)

    train_indices, val_indices = train_test_split(total_indices, test_size=0.1, random_state=42)

    train_generator = MemoryMappedDataGenerator(audio_filepath, video_filepath, indices=train_indices, batch_size=32, shuffle=True)
    val_generator = MemoryMappedDataGenerator(audio_filepath, video_filepath, indices=val_indices, batch_size=32, shuffle=False)

    # Fit the model
    training_history = vae_model.vae.fit(
        x=train_generator,
        validation_data=val_generator,
        epochs=500,
        callbacks=[early_stopping_callback, checkpoint_callback]
    )

    print('Training Completed')

    # Plot the training history
    Utils.plot_training_history(training_history)

    # Create the generator
    prediction_generator = PredictionDataGenerator(audio_train, video_train, batch_size=32)

    # Initialize an empty list to collect the predictions
    z_mean_list = []

    for i in range(len(prediction_generator)):
        sample = prediction_generator[i]
        print(f"Batch {i}:")
        print(f"Audio batch shape: {sample[0].shape}")
        print(f"Video batch shape: {sample[1].shape}")
        z_mean_batch, _, _ = vae_model.encoder.predict(sample)
        z_mean_list.append(z_mean_batch)

    # Concatenate all the batch-wise predictions into a single array
    z_mean = np.concatenate(z_mean_list, axis=0)

    # Apply t-SNE to the latent space representation
    tsne = TSNE(n_components=2, random_state=0)
    z_mean_2d = tsne.fit_transform(z_mean)

    # Plot the 2D visualization of the latent space
    plt.figure(figsize=(10, 8))
    plt.scatter(z_mean_2d[:, 0], z_mean_2d[:, 1], alpha=0.5)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('2D Visualization of the Latent Space')
    plt.show()

    # Load Test data
    video_data_test = Utils.load_data(f'{base_path}/input/test', video_processor, audio_processor)
    audio_test, video_test = Utils.prepare_data(video_data_test)

    audio_test, _, _ = Utils.normalize_audio(audio_test)
    video_test = Utils.normalize_video(video_test)

    x__, _ = vae_model.get_latent_space_pair_generative([audio_test, video_test])
    x__ = Utils.denormalize_audio(x__, a_min, a_max)

    # generate final output
    final = f'{output_path}/final'
    waveform_gen = audio_processor.features_to_audio(x__)
    audio_processor.save_audio(waveform_gen, f'{final}.wav')

    # Denoising the generated audio
    x_denoised = audio_processor.denoise_audio(x__)

    sf.write(f'{output_path}/final_enhanced.wav', x_denoised, 22050)

    print('\nAudio Spectogram (2D)\n')
    Utils.display_spectrogram(waveform_gen, title='Generated Spectrogram')

    # Generate final out video
    video_processor.frames_to_video(Utils.denormalize_video(video_test), f'{output_path}/final_enhanced.wav', f'{output_path}/final__out.mp4')
    Utils.render_video(f'{output_path}/final__out.mp4')



main()

