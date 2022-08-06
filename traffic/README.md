[Traffic](https://cs50.harvard.edu/ai/2020/projects/5/traffic/#traffic)
=======================================================================

Write an AI to identify which traffic sign appears in a photograph.

```
$ python traffic.py gtsrb
Epoch 1/10
500/500 [==============================] - 5s 9ms/step - loss: 3.7139 - accuracy: 0.1545
Epoch 2/10
500/500 [==============================] - 6s 11ms/step - loss: 2.0086 - accuracy: 0.4082
Epoch 3/10
500/500 [==============================] - 6s 12ms/step - loss: 1.3055 - accuracy: 0.5917
Epoch 4/10
500/500 [==============================] - 5s 11ms/step - loss: 0.9181 - accuracy: 0.7171
Epoch 5/10
500/500 [==============================] - 7s 13ms/step - loss: 0.6560 - accuracy: 0.7974
Epoch 6/10
500/500 [==============================] - 9s 18ms/step - loss: 0.5078 - accuracy: 0.8470
Epoch 7/10
500/500 [==============================] - 9s 18ms/step - loss: 0.4216 - accuracy: 0.8754
Epoch 8/10
500/500 [==============================] - 10s 20ms/step - loss: 0.3526 - accuracy: 0.8946
Epoch 9/10
500/500 [==============================] - 10s 21ms/step - loss: 0.3016 - accuracy: 0.9086
Epoch 10/10
500/500 [==============================] - 10s 20ms/step - loss: 0.2497 - accuracy: 0.9256
333/333 - 5s - loss: 0.1616 - accuracy: 0.9535

```

[Background](https://cs50.harvard.edu/ai/2020/projects/5/traffic/#background)
-----------------------------------------------------------------------------

As research continues in the development of self-driving cars, one of the key challenges is [computer vision](https://en.wikipedia.org/wiki/Computer_vision), allowing these cars to develop an understanding of their environment from digital images. In particular, this involves the ability to recognize and distinguish road signs -- stop signs, speed limit signs, yield signs, and more.

In this project, you'll use [TensorFlow](https://www.tensorflow.org/) to build a neural network to classify road signs based on an image of those signs. To do so, you'll need a labeled dataset: a collection of images that have already been categorized by the road sign represented in them.

Several such data sets exist, but for this project, we'll use the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) (GTSRB) dataset, which contains thousands of images of 43 different kinds of road signs.

[Understanding](https://cs50.harvard.edu/ai/2020/projects/5/traffic/#understanding)
-----------------------------------------------------------------------------------

First, take a look at the data set by opening the `gtsrb` directory. You'll notice 43 subdirectories in this dataset, numbered `0` through `42`. Each numbered subdirectory represents a different category (a different type of road sign). Within each traffic sign's directory is a collection of images of that type of traffic sign.

Next, take a look at `traffic.py`. In the `main` function, we accept as command-line arguments a directory containing the data and (optionally) a filename to which to save the trained model. The data and corresponding labels are then loaded from the data directory (via the `load_data` function) and split into training and testing sets. After that, the `get_model` function is called to obtain a compiled neural network that is then fitted on the training data. The model is then evaluated on the testing data. Finally, if a model filename was provided, the trained model is saved to disk.

The `load_data` and `get_model` functions are left to you to implement.

[Specification](https://cs50.harvard.edu/ai/2020/projects/5/traffic/#specification)
-----------------------------------------------------------------------------------

Complete the implementation of `load_data` and `get_model` in `traffic.py`.

-   The `load_data` function should accept as an argument `data_dir`, representing the path to a directory where the data is stored, and return image arrays and labels for each image in the data set.
    -   You may assume that `data_dir` will contain one directory named after each category, numbered `0` through `NUM_CATEGORIES - 1`. Inside each category directory will be some number of image files.
    -   Use the OpenCV-Python module (`cv2`) to read each image as a `numpy.ndarray` (a `numpy` multidimensional array). To pass these images into a neural network, the images will need to be the same size, so be sure to resize each image to have width `IMG_WIDTH` and height `IMG_HEIGHT`.
    -   The function should return a tuple `(images, labels)`. `images` should be a list of all of the images in the data set, where each image is represented as a `numpy.ndarray` of the appropriate size. `labels` should be a list of integers, representing the category number for each of the corresponding images in the `images` list.
    -   Your function should be platform-independent: that is to say, it should work regardless of operating system. Note that on macOS, the `/` character is used to separate path components, while the `\` character is used on Windows. Use [`os.sep`](https://docs.python.org/3/library/os.html) and [`os.path.join`](https://docs.python.org/3/library/os.path.html#os.path.join) as needed instead of using your platform's specific separator character.
-   The `get_model` function should return a compiled neural network model.
    -   You may assume that the input to the neural network will be of the shape `(IMG_WIDTH, IMG_HEIGHT, 3)` (that is, an array representing an image of width `IMG_WIDTH`, height `IMG_HEIGHT`, and `3` values for each pixel for red, green, and blue).
    -   The output layer of the neural network should have `NUM_CATEGORIES` units, one for each of the traffic sign categories.
    -   The number of layers and the types of layers you include in between are up to you. You may wish to experiment with:
        -   different numbers of convolutional and pooling layers
        -   different numbers and sizes of filters for convolutional layers
        -   different pool sizes for pooling layers
        -   different numbers and sizes of hidden layers
        -   dropout
-   In a separate file called *README.md*, document (in at least a paragraph or two) your experimentation process. What did you try? What worked well? What didn't work well? What did you notice?

Ultimately, much of this project is about exploring documentation and investigating different options in `cv2` and `tensorflow` and seeing what results you get when you try them!

You should not modify anything else in `traffic.py` other than the functions the specification calls for you to implement, though you may write additional functions and/or import other Python standard library modules. You may also import `numpy` or `pandas`, if familiar with them, but you should not use any other third-party Python modules. You may modify the global variables defined at the top of the file to test your program with other values.

<sub>*Assignment description above taken from https://cs50.harvard.edu/ai/2020/*</sub>

----

# Solution Process

## Video Demonstration
[<img src="https://i9.ytimg.com/vi/hhN4mqlNoU0/mq2.jpg?sqp=CNTxt5cG&rs=AOn4CLAB-GqvZfC6VIdayiyKuCAXGX41HA" width="50%">](https://www.youtube.com/watch?v=hhN4mqlNoU0)

## Goal
Optimise variable values by maximising accuracy but minimising the time taken for learning and prediction.

## Experimental Process
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

## Experimental Log
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

## Solution
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
