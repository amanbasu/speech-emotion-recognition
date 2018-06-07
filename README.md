# speech-emotion-recognition
Detecting human emotions using Deep Learning

In this project we will use Mel frequency cepstral coefficients (MFCC) to train a recurrent neural network (LSTM) and classify human emotions into happy, sad, angry, frustrated, sad, neutral and fear categories.

#### The dataset used is The Interactive Emotional Dyadic Motion Capture (IEMOCAP) collected by University of Southern California
the link for the same can be found [here](http://sail.usc.edu/iemocap/)

#### The dataset
The IEMOCAP database consists of 10 emotions. We selected the major 6 emotions viz. angry, neutral, frustrated, sad, excited and happy, in our training set. Features extracted from the raw audio of all sessions were saved along with their length and emotion.

To convert data into a consistent shape we have applied Bucket Padding. The data is first sorted according to their sequence lengths and then divided into a specific number of buckets. The length of data thus divided is in close range of each other which eliminates extra padding. This method is used in Bucket Iterator which is used to get the batch if desired examples.

For selecting a batch, a bucket is chosen at random containing sorted data, out of that bucket contiguous examples equal to the batch size are chosen. The examples are padded to the shape of maximum sequence length and then shuffled. This gives the desired batch.
the code for bucket iterator is taken from [R2RT](https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html)



