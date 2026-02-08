# AI_Club_Task
Name : Yatish Piplani
ID No. : 2025A4PS1341P

The user has to just write the name of the audio file in the command prompt when running the file and the program takes it as the .wav input and provides the output as emotion and the confidence with which it is predicting the emotion.

P.S: The best model weights are saved in the .keras file. They are not the ones displayed in the jupiter notebook.

#What Project does
This project takes an audio speech file as an input, reads it, converts it to a log-mel spectrogram as that is what humans percieve sound as. It analyzes the log-mel spectrogram and classifies them using a 2D convolution neural network to identify human emotions.

#Preprocessing
The model takes the audio file,resamples all audios to 22050Hz trims the leading and trailing silence, shifts pitch by 2 semitones and injects noise to make it more real-life scenario like. It also performs padding on each file to keep the training dataset uniform. Validation and test sets were kept unaugmented.

#Model
2D Convolutional neural network with ReLU activation. It further performs batch normalization and max pooling to filter out data. Dropout feature is used so that the model does not memorize any audio or voices. A softma output layer with 8 units is produced.

Data split : 80% Train/10% Valuation/ 10% Test
Epochs : 20
Batch Size : 64
Loss Function : Categorical cross entropy
The Best performing model weights were saved based on the best valuation loss-based checkpoints.

#Evaluation Metrics
Macro-F1 score - 0.36
The confusion matrix generated represents how many audios of each emotion were detected during testing.
Accuracy - ~35%

#Bias Evaluation
A male vs female macro f1 score comparison is done to check if the model performs or is bias towards a specific gender. It was observed that for male it was 0.28 while for females 0.31 which show there is a very minimal difference and the model does not show any bias towards any one gender.
