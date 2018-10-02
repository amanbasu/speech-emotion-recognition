# Recognising Human Emotions From Raw Audio
Collaborator: [Aman Agarwal](https://amanbasu.github.io), [Aditya Mishra](https://aditya985.github.io)

In this project we will use Mel frequency cepstral coefficients (MFCC) to train a recurrent neural network (LSTM) and classify human emotions into happy, sad, angry, frustrated, sad, neutral and fear categories.

#### The dataset used is The Interactive Emotional Dyadic Motion Capture (IEMOCAP) collected by University of Southern California
the link for the same can be found [here](http://sail.usc.edu/iemocap/)

#### The dataset
The IEMOCAP database consists of 10 emotions. We selected the major 6 emotions viz. angry, neutral, frustrated, sad, excited and happy, in our training set. Features extracted from the raw audio of all sessions were saved along with their length and emotion. We used the first 20 mfcc coefficients as the feature vector, the process can be found in [notebook](https://github.com/amanbasu/speech-emotion-recognition/blob/master/create_mfcc.ipynb)

To convert data into a consistent shape we have applied Bucket Padding. The data is first sorted according to their sequence lengths and then divided into a specific number of buckets. The length of data thus divided is in close range of each other which eliminates extra padding. This method is used in Bucket Iterator which is used to get the batch if desired examples.

For selecting a batch, a bucket is chosen at random containing sorted data, out of that bucket contiguous examples equal to the batch size are chosen. The examples are padded to the shape of maximum sequence length and then shuffled. This gives the desired batch.
the code for bucket iterator is taken from [R2RT](https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html)

#### Model
We used two layers of Bidirectional LSTM followed by attention in the last layer. The batch size was kept as 128 with the learning rate of 1e-4.

#### Results
The model was trained for 500 epochs and after which the curve almost reached a plateau. The model showed overfitting when the dropout was not used. We then applied a dropout of keep probability 0.8 between the last LSTM layer and the output layer.

Adding dropout reduced the overfitting of the model and increased its overall accuracy. The model showed an unweighted accuracy across six emotions of 45% with the validation accuracy of 42%.

Dropout of 0.2             |  No Dropout
:-------------------------:|:-------------------------:
![](https://github.com/amanbasu/speech-emotion-recognition/blob/master/plot_dropout.png)  |  ![](https://github.com/amanbasu/speech-emotion-recognition/blob/master/plot_no_dropout.png)

### Tensorflow model
Tensorflow implementation of the model has been added. The repository contains two files, [speech_emotion_gpu](https://github.com/amanbasu/speech-emotion-recognition/blob/master/speech_emotion_gpu.py) to run the model on gpu and [speech_emotion_gpu_multi](https://github.com/amanbasu/speech-emotion-recognition/blob/master/speech_emotion_gpu_multi.py) which makes the file run parallelly on multiple gpus.

### Input data for model can be downloaded from the [link](https://drive.google.com/file/d/1QidPJVsdUnYXj0VAGIrffmDl3pjA6RLl/view?usp=sharing).
It consists of the following features: F0 (pitch), voice probability, zero-crossing rate, 12-dimensional Mel-frequency cepstral coefficients (MFCC) with log energy, and their first time derivatives. The features have been taken from [this](https://www.microsoft.com/en-us/research/publication/high-level-feature-representation-using-recurrent-neural-network-for-speech-emotion-recognition/) paper.

