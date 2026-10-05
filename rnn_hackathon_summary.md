# RNN Hackathon Assignment Summary

## Project Overview
This assignment focused on building an LSTM-based classifier for the sklearn Digits dataset. The task was to classify handwritten digits from 0 to 9, with a baseline accuracy of 10% and a target accuracy of 93%.

## Dataset
- Dataset: sklearn Digits
- Samples: 1,797 total images
- Image size: 8 x 8 pixels
- Classes: 10 digits (0–9)
- Data source: built into scikit-learn, so no internet download was needed

The dataset was treated as a sequence problem because each image was viewed as 8 time-steps, with each row of 8 pixels representing one step. This allowed the model to read the image row-by-row, top to bottom, like a sequence.

## Methodology
1. Loaded the Digits dataset and normalized pixel values to the range [0, 1].
2. Split the data into an 80/20 train/test set using stratification to preserve class balance.
3. Converted the data into PyTorch tensors and created DataLoaders.
4. Built an LSTM model with:
   - Input size: 8
   - Hidden size: 128
   - Number of layers: 2
   - Dropout: 0.3
   - Output: 10 class logits
5. Used CrossEntropyLoss as the loss function and Adam as the optimizer.
6. Applied learning-rate scheduling and gradient clipping to improve stability and convergence.

## Model Architecture
The model followed an LSTM → dropout → linear architecture:
- Input sequence shape: (batch, 8, 8)
- LSTM processes each of the 8 rows as sequential time-steps
- Final hidden state from the last layer is passed to a linear classifier
- Output is a vector of 10 logits representing the digit classes

## Training and Evaluation
The model was trained for 50 epochs. During training, the loss decreased steadily, and both training and test accuracy improved over time. The final evaluation was performed on the hold-out test set, and a confusion matrix was used to inspect misclassifications.

## Results
The final model achieved a test accuracy at or above the 93% target, which is well above the 10% random/majority-class baseline.

### Key outcome
- Baseline: 10%
- Target: 93%
- Model result: 93%+ accuracy

This demonstrates that the LSTM was effective for digit recognition when the images were represented as row-based sequences.

## What Worked Well
- Treating each digit image as a sequence of rows was a natural fit for an LSTM.
- The 2-layer LSTM with hidden size 128 learned the patterns well.
- Adam optimizer and scheduled learning rate helped training converge effectively.
- Gradient clipping prevented unstable training behavior.

## Challenges and Fixes
Early training runs suffered from unstable loss spikes, which were reduced by adding gradient clipping. A higher dropout value was also found to be too aggressive for the dataset, so it was lowered to 0.3 for better performance.

## Conclusion
This assignment showed that recurrent neural networks, specifically LSTMs, can be successfully applied to image classification tasks when the spatial structure is represented as a sequence. The model not only outperformed the random baseline but also reached the assignment target, confirming the effectiveness of the approach.
