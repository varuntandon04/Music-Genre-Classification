# Feature Extraction From Music Genres

Feature exrtaction is the one of the first steps in the ML pipeline that can dictate how well a model can perform. Furthermore, feature extraction from audio is vital as models won't be able to learn well enough from the sound clips alone. Thus in this project, we see how we can utilize the ```librosa``` library to extract important information from audio.

We also will go through different methods of visualizing sound, and focusing on Mel Spectrography as the primary visualization tool to depict audio data. These visualizations can be key in using CNNs to classify different genres using these spectrograms. 

The data was taken from the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). This dataset contains the mel spectrograms and features extracted from the audios of a thousand tracks comprising of ten genres. In genre_feature_extraction.ipynb, the process of forming the spectrograms and the feature extraction is highlighted. 

Some of the functions used were:

* `librosa.feature.chroma_stft`: Computes a chromagram from a waveform or power spectrogram.
* `librosa.feature.rms`: Computes root-mean-square (RMS) value for each frame, either from the audio samples or from a spectrogram
* `librosa.feature.spectral_centroid`: Computes the spectral centroid.
* `librosa.feature.spectral_bandwidth`: Computes p’th-order spectral bandwidth.
* `librosa.feature.spectral_rolloff`: Computes roll-off frequency.

Which resulted in 33 extracted features.


Here are a few examples of a mel spectrogram:

<img src='img/mel spectrogram example.png'>

