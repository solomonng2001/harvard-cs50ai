qwerqwere

----

# Goal
Optimise variable values by maximising accuracy but minimising the time taken for learning and prediction.

# Experimental Process
Experiment begins with the following variables set to the following values. At one time, only one variable is changed to observe the effects of only one variable, while all else remains constant, allowing for comparison between various experiments/set-ups.

```python
convolutional_pooling_layers = 1
filter_number = 16
filter_size = (2, 2)
max_pool_size = (2, 2)
hidden_layers = 1
hidden_size = 16
dropout = 0.8
```

To explore the effects of variables efficiently, the values of variables were kept to be computationally light initially, such that less time is taken to experiment.

# Experimental Log
Through trial and error, changes that work are retained, as various inferences/observations are recorded below.

Experiment No. | Convolutional & Pooling Layers | No. of Filters | Size of Filters | Size of Pooling Layers | No. of Hidden Layers | Size of Hidden Layers | Droopout | Data | Observations/Inference
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
1 | 1 | 16 | (2, 2) | (2, 2) | 1 | 16 | 0.3 | 333/333 - 8s - loss: 3.4937 - accuracy: 0.0540 | 
2 | 1 | 16 | (2, 2) | (2, 2) | 1 | 16 | 0.9 | 333/333 - 4s - loss: 3.5044 - accuracy: 0.0562 | Increasing dropout ratio increases accuracy and decreases time taken (compared with experiment 1).
3 | 3 | 16 | (2, 2) | (2, 2) | 1 | 16 | 0.9 | 333/333 - 6s - loss: 3.5052 - accuracy: 0.0569 | Cannot increase number of convolutional and pooling layers beyond 3, due to "ValueError: Negative dimension size". Minimal increase in accuracy with minimal increase in time (compared with experiment 2).
4 | 3 | 16 | (4, 4) | (2, 2) | 1 | 16 | 0.9 | 333/333 - 6s - loss: 3.4999 - accuracy: 0.0553 | Negative effect on accuracy (compared with experiment 3). Filter size cannot be more than (4, 4), due to "ValueError: Negative dimension size".
5 | 3 | 16 | (2, 2) | (3, 3) | 1 | 16 | 0.9 | "ValueError: Negative dimension size" | Size of pooling layer cannot be more than (2, 2)
6 | 3 | 16 | (2, 2) | (2, 2) | 1 | 64 | 0.9 | 333/333 - 4s - loss: 3.5040 - accuracy: 0.0557 | Negative effect on accuracy (compared with experiment 3). Hypothesis: Dropout must be decreased together with increase in size of hidden layer, for greater degree of learning with a more complex hidden layer.
7 | 3 | 16 | (2, 2) | (2, 2) | 1 | 64 | 0.5 | 333/333 - 3s - loss: 0.9799 - accuracy: 0.6956 | Significant increase in accuracy (compared with experiment 6). Hypothesis in experiment 7 is true (also need to take into account the limitations of a low dropout ratio stated in experiment 2). Hypothesis: Jointly increasing number of hidden layers and size of hidden layers is required for positive effect on accuracy (to increase the complexity of the hidden layers for learning and prediction).
8 | 3 | 16 | (2, 2) | (2, 2) | 3 | 64 | 0.5 | 333/333 - 3s - loss: 0.6344 - accuracy: 0.7853 | Significant increase in accuracy (compared with experiment 7). Hypothesis in experiment 8 is true.
9 | 3 | 32 | (2, 2) | (2, 2) | 3 | 64 | 0.5 | 333/333 - 8s - loss: 0.4028 - accuracy: 0.8739 | Significant increase in accuracy and time taken. Having explored the effects of each variable so far, we attempt to modify variables further to increase accuracy and reduce time taken.

Further optimisation of variable's values increased accuracy, while reducing time taken. Additional observations were made in the process. Reducing the number of filters from 32 to 16 helped to greatly reduce the time taken for both training and prediction. The focus was then on increasing variables, such as number (from 3 to 4) and size (from 64  to 128) of hidden layers, that have an effect on increasing accuracy whilst maintaining a low duration of learning and prediction. Reducing the dropout ratio below 0.5 produced better accuracy (possiblly allowing greater degree of learning), but at the expense of inconsistent accuracy. Reducing the number of convolutional and pooling layers from 3 to 2 also increased accuracy. Increasing filter size beyond (2, 2), had a disastrous effect on accuracy.

# Solution
Following are the optimal variable values.

```python
convolutional_pooling_layers = 2
filter_number = 16
filter_size = (2, 2)
max_pool_size = (2, 2)
hidden_layers = 4
hidden_size = 128
dropout = 0.5
```

Data: 333/333 - 5s - loss: 0.3315 - accuracy: 0.9183

# Video Demonstration
The functionality of the machine learning algorithm is demonstrated in the video link below.<br>
<a href="https://youtu.be/hhN4mqlNoU0" target="_blank">
    <img src="https://i9.ytimg.com/vi/hhN4mqlNoU0/mq2.jpg?sqp=CNiX3pIG&rs=AOn4CLCrG8fio4KR1-anMd5PrlQjdiX50g" width="240" height="180" border="10" />
</a>
