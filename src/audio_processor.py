import librosa
import numpy as np
import soundfile as sf
import subprocess
import tempfile
import os

class AudioProcessor:
    def __init__(self, sr=22050, n_fft=2048, hop_length=735):
        """
        Initialize the AudioProcessor with specific audio processing parameters.
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_feature(self, audio_path):
        """
        Extracts the spectrogram from an audio file.
        """
        try:
            # Load the audio file with librosa
            audio, sr = librosa.load(audio_path, sr=self.sr)
            # Compute the Short-Time Fourier Transform (STFT)
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            # Convert the amplitude spectrogram to dB-scaled spectrogram
            features = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
            # Return the features, excluding the Nyquist frequency bin
            return features[:-1].T
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return None

    def features_to_audio(self, features_batch):
        """
        Reconstructs audio waveforms from a batch of spectrogram features.
        """
        concatenated_waveform = []
        for features in features_batch:
            # Re-pad the Nyquist frequency bin with zeros to match original spectrogram shape
            S = np.pad(features.T, ((0, 1), (0, 0)), mode='constant', constant_values=0)
            # Convert the dB-scaled spectrogram back to a linear amplitude spectrogram
            S_linear = librosa.db_to_amplitude(S)
            # Invert the spectrogram to a waveform using the Griffin-Lim algorithm
            waveform = librosa.griffinlim(S_linear, n_iter=32, hop_length=self.hop_length, win_length=self.n_fft, n_fft=self.n_fft)
            concatenated_waveform.append(waveform)

        # Concatenate all waveforms in the batch into a single waveform
        concatenated_waveform = np.concatenate(concatenated_waveform, axis=0)
        return concatenated_waveform

    def denoise_audio(self, feature_batch):
        """
        Denoises an audio waveform using spectral gating.
        """
        audio_waveform = self.features_to_audio(feature_batch)
        # Compute the Short-Time Fourier Transform (STFT) of the audio
        S = librosa.stft(audio_waveform, n_fft=self.n_fft, hop_length=self.hop_length)
        # Convert the complex-valued STFT to magnitude and phase components
        magnitude, phase = librosa.magphase(S)
        # Estimate the noise profile by averaging the magnitude of the first 20 frames,
        # assuming these frames mostly contain noise
        noise_profile = np.mean(np.abs(S[:, :20]), axis=1)
        # Define a threshold for spectral gating; values below this threshold are considered noise
        threshold = 1.5 * noise_profile
        # Apply spectral gating: set magnitudes below the threshold to 0 to suppress noise
        magnitude[magnitude < threshold[:, None]] = 0
        # Reconstruct the denoised STFT by combining the modified magnitude with the original phase
        S_denoised = magnitude * phase
        # Perform the inverse STFT to convert back to an audio time series
        x_denoised = librosa.istft(S_denoised, hop_length=self.hop_length, win_length=self.n_fft)
        # Boost volume
        x_denoised = x_denoised * 10
        # Ensure the waveform doesn't exceed the -1.0 to 1.0 range to prevent clipping
        x_denoised = np.clip(x_denoised, -1.0, 1.0)
        return x_denoised

    def save_audio(self, waveform, file_path):
        """
        Saves an audio waveform to a file.
        """
        # Write the waveform to the file using the specified sample rate
        sf.write(file_path, waveform, self.sr)

    def extract_audio_from(self, input_video_path):
        """
        Extracts the audio track from a video file and saves it as an mp3 file.
        """
        try:
            # Generate a temporary file path for the extracted audio
            extracted_audio_path = tempfile.mktemp(suffix='.mp3')
            # Use ffmpeg to extract the audio track from the video
            subprocess.run(['ffmpeg', '-i', input_video_path, '-q:a', '0', '-map', 'a', extracted_audio_path], check=True)
            return extracted_audio_path
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio from video: {e}")
            return None
