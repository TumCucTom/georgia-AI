![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![logo](images/logo.jpg)
# Georgia-AI
[How to use](#setup-your-venv) | [Design and review](#performance)

A machine learning AI that looks at a set of wordle results for a given day and tries to infer that days word given how people guessed.

## Overview
We give our bot an input representing several wordle results (the format shown below). We return a probability ditributed over the five letter words, giving the most likely words for that day.

```angular2html
Day 999 5/6
⬜⬜⬜🟨🟨
🟨🟨⬜⬜⬜
⬜🟩🟩🟩🟩
⬜🟩🟩🟩🟩
🟩🟩🟩🟩🟩
```

## Setup your venv:
Download python 3.7 if you only have higher versions installed. Or use your older version, so long as it is supported by pytorch
```angular2html
python3.7 -m venv env3.7
source evn3.7/bin/activate
```
then in your env
```angular2html
pip3 install torch
```

## Processing the data

If you want to add to the data or replicate the process, here are the steps:

- Export whatsapp chat without media
- Rename the file to all-messages and put it under ```/data```
- You can now use the processing scripts to change you data depending on what model you wish to use
- Each script has a docstring explaining what it will do to the data where it will be outputted and if a second file is needed
- You can find the 5757 5 letter words and all wordle answers up to 12/12/2024 in the resource folder

## Using georgia-AI
Train one of the models under model/train. Find out more on these in the [train](#training) area.

Add your input in the correct format for the model you trained in `data/input.csv`.

Run georgia-AI with:
```
cd model
source env3.7/bin/activate
python3 georgia-AI.py
```
or
```
cd model
source env3.7/Scripts/activate
python3 georgia-AI.py
```
You will find the results in `data/results`

## Performance
### Overview
When considering correctness of words as a whole, we can only talk about the performance after beam and the use a dictionary have been used. This is 28% accuracy. This is very impressive considering the 28.72% letter by letter accuracy directly after the model is run.

The actual performance of the model is very poor however, with 28% accuracy.
### Speculation
We can look at ```data/results/beam-serach-dict.txt``` to see that:
- obscure endings are dealt with well: i.e twist
-Terrible with words that start with ‘s’. It always tries to guess words like ‘stare’ ‘stale’ ‘stole’. 
  - Probably as a product of those being the underlying words for first guesses for most people, so makes up a large portion of the underlying patterns in the data. 
- It also struggles differentiating between word endings where people tend to take a lot of guess with the same ending ‘ly’ ’ch’ ‘ck’ ‘dy' etc.
## Input and Output
Before using one variable for whether each letter of each guess is each colour, I use one variable for each letter of every guess only, with 0 representing black or white, 1 representing yellow and 2 representing green. This produced around 17% accuracy.

The output of the Wordle AI is a probability distribution over all 5-letter words, and it's one-hot encoded to make it easier to work with. Instead of directly outputting a list of 5757 possible words, the model represents the probability of each word using a vector, where each element corresponds to a word in the list. This one-hot encoding allows for a more compact and efficient way of handling the output, making it quicker to compute and easier to integrate with the model’s decision-making process.

One big advantage of this approach over the extensive 5757-word list is that it simplifies the calculations. Instead of needing to manage the entire list at once, the one-hot encoding lets the model focus on the relevant probabilities and make updates more easily after each guess. Plus, it reduces the complexity of the output space, which can help in speeding up the learning process and improving the AI’s ability to predict the correct word more efficiently.

The one-hot with larger input gave 28% and extensive 22%. We can softmax then argmax over the output to get the most likely letters but that's not entirely useful so we use beam search [later](#processing-the-data)

## The model
### One-hot
The network begins with three fully connected (linear) layers: the first layer maps 900 input features (likely derived from 10 sets of guesses) to 2048 neurons, the second layer reduces this to 1024 neurons, and the third layer further reduces it to 512 neurons. The decision to increase the number of neurons in the first layer (to 2048) is driven by the need to extract complex features from the input data. Given the large amount of information that needs to be learned from the 10 sets of guesses, a larger layer size helps the model capture more intricate relationships in the data. The subsequent layers progressively reduce the number of neurons, which allows the network to distill the most important features while reducing computational complexity. This design helps the network learn increasingly abstract representations of the input.

The final layer of the network is a fully connected layer with 130 neurons, which corresponds to 26 possible letter predictions for each of the 5 positions in a word. There is no activation function applied here, as softmax will be used during the loss calculation to convert the logits into probabilities. This output layer is structured in this way to accommodate the task of predicting a word’s individual letters, where each position in the 5-letter word is classified as one of 26 letters from the English alphabet.

To improve training efficiency and stability, batch normalization is applied after each of the first three layers. This technique helps stabilize the learning process by normalizing the inputs to each layer, reducing the internal covariate shift that often occurs during training. By normalizing the activations, batch normalization enables the model to train faster and achieve better generalization. It’s applied to the 2048, 1024, and 512 neuron layers, each of which has progressively smaller sizes, helping to maintain consistency in the distribution of activations as the data moves through the network.

To prevent overfitting, dropout with a rate of 0.4 is applied after each hidden layer. Dropout forces the model to learn more robust representations by randomly setting a fraction of the input units to zero during each forward pass. This regularization technique reduces the risk of the model becoming overly reliant on specific neurons, ensuring that the learned features generalize well to new data and reducing the potential for overfitting.

The model uses ReLU (Rectified Linear Unit) as the activation function for all hidden layers. ReLU is widely used because it effectively mitigates the vanishing gradient problem, which can slow down training when using other activation functions like sigmoid or tanh. By allowing gradients to propagate more effectively, ReLU enables faster and more stable learning while still capturing non-linear relationships in the data.

Finally, the output of the network is reshaped using the `view` method to produce a tensor of dimensions `(-1, 5, 26)`, where 5 corresponds to the number of positions in the word, and 26 corresponds to the possible letter choices at each position. This reshaping is crucial for the word prediction task, as it structures the output in a way that aligns with the goal of classifying each position of the word independently.

### Extensive
#### 1. **Fully Connected Layers and Layer Normalization**

The model starts with a fully connected (linear) layer, `fc1`, which takes in 900 input features (presumably representing multiple sets of guesses) and transforms them into 2048 units. Following this, a layer normalization (`ln1`) is applied to stabilize training by normalizing the activations and reducing the effects of internal covariate shift. The next block, `fc2`, further reduces the dimensionality from 2048 to 1024, with another layer normalization (`ln2`) to ensure smooth training dynamics. This approach of gradually reducing the number of units in successive layers allows the model to first capture rich features from the input data and then distill those features into more compact representations.

Similarly, the model continues with additional blocks where each layer (`fc3`, `fc4`) further reduces the dimensionality. Specifically, `fc3` reduces from 1024 to 512, and `fc4` reduces from 512 to 256, with corresponding layer normalizations (`ln3` and `ln4`) applied to each layer. The gradual reduction in dimensionality helps the model progressively focus on more abstract representations of the input, facilitating better learning of underlying patterns.

#### 2. **Skip Connections and Residual Connections**

A key feature of this model is the use of skip or residual connections between layers. In the second block, after processing the output of `fc2`, a projection layer `fc_proj` is used to project the output of the first block (`x1`) from 2048 units to 1024 units, matching the output size of `x2`. The output from `x2` is then added to this projected value, creating a residual connection. This allows the model to retain and propagate information from earlier layers, which has been shown to help in training deeper networks by alleviating the vanishing gradient problem and improving the flow of gradients during backpropagation.

Similarly, in the fourth block, after processing through `fc4`, the output of the third block (`x3`) is projected to 256 units using the `fc_proj2` layer, and added to the output of `x4` as a residual connection. These residual connections help ensure that the network can learn more complex features without losing useful information from earlier layers, improving both convergence speed and model performance.

#### 3. **Dropout**

Dropout is applied after each hidden layer to prevent overfitting. A dropout rate of 0.2 means that 20% of the neurons are randomly dropped during each forward pass, forcing the network to learn more robust and generalizable features. By randomly deactivating neurons during training, dropout reduces the model's reliance on specific neurons and encourages the development of multiple independent paths for prediction. This technique is particularly useful in preventing overfitting, especially in models with many parameters.

#### 4. **Swish Activation**

The model uses the Swish activation function (denoted as `SiLU` in PyTorch) instead of traditional activation functions like ReLU. Swish has been found to outperform ReLU in many cases because it allows for smoother gradients, especially for deeper networks. The smooth, non-monotonic nature of Swish helps improve optimization by reducing the likelihood of dead neurons (a problem associated with ReLU) and ensuring a more stable gradient flow throughout the network. This contributes to the model’s ability to learn more effectively, particularly in complex tasks like word prediction.

#### 5. **Output Layer**

The output layer, `fc_out`, is a fully connected layer that maps the final representation (of size 256) to 5757 units, corresponding to the size of the output vocabulary. This large number of units likely corresponds to the total number of possible words or tokens in the model's vocabulary. The output of this layer represents the logits, which are later converted to probabilities using softmax during the loss computation. This layer is crucial for the final classification task, where each unit corresponds to the likelihood of a specific word or token.

#### 6. **Weight Initialization**

The model employs He initialization (via `kaiming_uniform_`) for the weight matrices of all fully connected layers. He initialization is particularly effective when using ReLU-based activations, as it ensures that the variance of activations remains consistent across layers. This initialization technique helps prevent issues with vanishing or exploding gradients, which can severely hinder training. The biases of the layers are initialized to zeros, which is a common and effective strategy.

## Training

### One-hot
- **Smooth Cross-Entropy Loss**: This loss function introduces label smoothing to regularize the model, improving generalization.
- **Batch Normalization**: Batch normalization layers are used to stabilize training and reduce the impact of internal covariate shifts.
- **Dropout**: A dropout rate of 0.4 helps prevent overfitting, particularly given the complexity of the model.
- **Learning Rate Scheduling**: The cyclic learning rate scheduler allows for better convergence by adjusting the learning rate during training.
- **Accuracy Calculation**: The model's accuracy is evaluated based on how well it predicts each letter in the 5-letter word, ensuring fine-grained evaluation.
- **TorchScript Export**: The trained model is exported for use in production environments without relying on Python.
#### 1. **Dataset Preparation**

The dataset is loaded from a CSV file using NumPy's `np.loadtxt` function, where the first 900 columns are used as input features (`X`), and the last 130 columns represent the output (`y`), which corresponds to a 5-letter word with 26 possible letter classes for each position (i.e., 5 positions, each having 26 letters). The input data `X` is converted into a PyTorch tensor of type `float32`, while the target labels `y` are converted into a tensor of type `long` (for use with the `CrossEntropyLoss`).

#### 2. **Smooth Cross-Entropy Loss Function**

A custom loss function, `smooth_cross_entropy_loss`, is defined to implement label smoothing. Label smoothing helps regularize the model by softening the target labels, which in turn prevents overfitting and improves generalization. In the function:
- The one-hot encoded labels are adjusted by the smoothing factor (`smoothing=0.1`).
- The predicted logits are passed through `log_softmax`, and the loss is computed by calculating the cross-entropy between the predicted and the smoothed target labels.
- This loss is then averaged over the batch, which helps in the model training process by allowing smoother gradients.

#### 3. **Model Architecture**

The `ImprovedWordPredictor` model is designed with four fully connected layers (`fc1` to `fc4`) and includes batch normalization layers (`bn1`, `bn2`, `bn3`) to stabilize training. The model incorporates:
- **Dropout**: A dropout layer with a rate of 0.4 is used after each hidden layer to prevent overfitting and improve generalization.
- **ReLU Activation**: ReLU is applied after each hidden layer to introduce non-linearity, which allows the model to capture complex patterns in the data.
- The output of the network (`fc4`) has a shape of `[batch_size, 130]`, representing the 5 positions (for each letter) and 26 possible classes (letters). The output is reshaped to match this 5x26 format, suitable for multi-class classification over multiple positions in the word.

#### 4. **Training Setup**

- **Loss Function**: The model uses the previously defined smooth cross-entropy loss.
- **Optimizer**: The Adam optimizer (`optim.Adam`) is used with a learning rate of `0.0005` and weight decay of `1e-5` for L2 regularization. The optimizer helps update the model's parameters efficiently during backpropagation.
- **Learning Rate Scheduler**: A **CyclicLR scheduler** is used to adjust the learning rate during training. The learning rate oscillates between a minimum (`base_lr=0.0001`) and maximum (`max_lr=0.001`) value, helping the model escape local minima and achieve better convergence.

#### 5. **Training Loop**

The model is trained over `100` epochs with a batch size of `32`. For each epoch:
- The model is set to training mode (`model.train()`), and mini-batches of size 32 are fed into the model.
- The optimizer gradients are zeroed out at the beginning of each batch using `optimizer.zero_grad()`.
- The batch is passed through the model to get predictions (`y_pred`).
- The target labels (`ybatch`) are reshaped to represent 5 positions per word (each with 26 possible letter classes). The target labels are then flattened into a 1D tensor (`ybatch_reshaped`), which matches the shape of the predicted logits (`y_pred_reshaped`).
- The loss is computed using the smooth cross-entropy loss function.
- Backpropagation is performed to compute gradients (`loss.backward()`), and the optimizer updates the model's weights (`optimizer.step()`).
- The learning rate scheduler is stepped (`scheduler.step(loss)`) to adjust the learning rate based on the loss value.
- The loss for each epoch is printed to track training progress.

#### 6. **Evaluation and Accuracy Calculation**

After training, the model is evaluated on a separate test dataset. The test dataset is loaded and split into input features (`X`) and output labels (`y`). The input features are passed through the model in evaluation mode (`model.eval()`), and the predictions are computed.

The predicted labels are compared with the actual labels, with accuracy being calculated as the percentage of matching letters across all 5 positions of the words. The comparison is done by:
- Applying `argmax` on both the predicted and actual labels to get the predicted letter indices for each position.
- Calculating the accuracy by comparing the predicted indices with the actual ones and computing the mean accuracy.

#### 7. **Exporting the Model**

After the model is trained and evaluated, it is exported to **TorchScript** format using `torch.jit.script`, which allows the model to be deployed and used in production environments without needing the full PyTorch framework. The scripted model is saved as `model_scripted.pt`.



### Extensive
- **Mini-batch training**: The dataset is divided into mini-batches for efficient gradient computation and to ensure better generalization by avoiding overfitting.
- **Dropout**: Dropout with a rate of 0.2 is applied during training to regularize the model, ensuring it doesn't rely too heavily on specific neurons, thus improving generalization.
- **Residual connections**: Skip connections between layers help preserve information, allowing deeper networks to train more effectively.
- **Swish activation**: Swish is used instead of ReLU for smoother gradient flow, potentially improving the model's training dynamics.
- **Cross-entropy loss**: This loss function is used for multi-class classification, making it appropriate for the task of predicting one of 5757 word classes.
- **Learning rate scheduling**: The reduced learning rate ensures stable training and prevents the model from diverging.
#### 1. **Dataset Preparation**

The dataset is loaded using NumPy's `np.loadtxt` function, where the features (`X`) are extracted from the first 900 columns, and the target labels (`y`) are the class indices (the corresponding 5-letter word classes, with a total of 5757 possible classes). These arrays are then converted to PyTorch tensors (`X` as `float32` and `y` as `long`) to be compatible with the neural network model. Converting the dataset to tensors is essential because PyTorch models require tensor inputs for training.

#### 2. **Model Definition and Initialization**

The `WordPredictor` model is defined as a subclass of `nn.Module` with four main blocks of fully connected layers, each accompanied by layer normalization, activation functions (Swish), dropout for regularization, and residual (skip) connections. The model's architecture includes multiple hidden layers, with residual connections helping to preserve information and mitigate potential issues with vanishing gradients in deep networks. The model's output layer, `fc_out`, produces logits for each of the 5757 possible word classes.

The model weights are initialized using He initialization (via `kaiming_uniform_`), which is commonly used for layers with ReLU or Swish activations. This helps ensure proper gradient flow during the early stages of training, allowing the model to learn effectively.

#### 3. **Loss Function and Optimizer**

The training process uses the **cross-entropy loss** function (`nn.CrossEntropyLoss`), which is well-suited for classification tasks with a large number of classes. This loss function computes the difference between the predicted logits and the true class indices (`y`). The optimizer used is **Adam** (`optim.Adam`), a popular choice for training deep neural networks due to its adaptive learning rate and ability to handle sparse gradients. The learning rate is set to `0.0005`, which has been reduced to ensure more stable training and prevent overshooting during optimization.

#### 4. **Training Loop**

The model is trained for `50` epochs, with each epoch consisting of multiple mini-batches of size `10`. The dataset is iterated in batches, and for each batch:

- **Zero gradients**: `optimizer.zero_grad()` clears the previous gradients stored in the model's parameters.
- **Forward pass**: The batch of inputs (`Xbatch`) is passed through the model to generate predictions (`y_pred`).
- **Loss calculation**: The loss is computed using the predicted outputs and the true labels (`ybatch`).
- **Backward pass**: The gradients are computed by calling `loss.backward()`, which propagates the error back through the network.
- **Optimizer step**: The optimizer updates the model's weights using `optimizer.step()`, which applies the computed gradients to minimize the loss.

After each epoch, the loss for the last mini-batch is printed, allowing the user to track the progress of training.

#### 5. **Testing the Model**

Once training is complete, the model is evaluated on a separate test set. The test set is loaded from a CSV file, and the input features (`X_test`) and true labels (`y_test`) are extracted and converted to PyTorch tensors.

During evaluation, the model's predictions are obtained by performing a forward pass through the model with `torch.no_grad()` (to disable gradient computation, saving memory and computation during inference). The predicted logits are converted to class indices by calling `argmax(dim=1)`, which selects the class with the highest predicted probability.

**Letter-level accuracy** is then computed by comparing each predicted word to the actual word from the test set. For each word, the number of matching letters is counted, and the total accuracy is calculated as the ratio of correctly predicted letters to the total number of letters in the dataset. This provides a more granular evaluation of the model's performance, focusing on how well it predicts individual letters within words.
