# Lecture 1

## Lecture 1: Getting Started

In Neural Networks, you don’t provide the model features. You ask it to learn them.

Deep learning is called deep learning because we can take the initial features the model learns and combine them to create more advanced features.

Neural net is the basic function in deep learning

fast ai is a library built on top of pytorch

fastest, widest used computer vision model is resnet18 … never heard of it!

- takes under 10 sec to train on a GPU

"Fine-tuning" a model means that we're starting with a model someone else has trained using some other dataset (called the *pretrained model*), and adjusting the weights a little bit so that the model learns to recognize your particular dataset

“Learner” is something that combines a model and the data to train the model with.

Segmentation:

- where we take photos and we color in every pixel according to what it represents
- another way computer vision is used

Collaborative filtering - recommendation systems:

- which users liked which products, predict what to show these users by comparing similar users and what those users have liked
- similar in the sense of people who have liked the same types of products

```python
from fastai.collab import *
path = untar_data(URLs.ML_SAMPLE)
dls = CollabDataLoaders.from_csv(path/'ratings.csv')
dls.show_batch()
```

```python
from fastai.vision.all import *
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```

- no such thing as a pre-trained collaborative filtering model

First neural net created in 1957, Mark 1 Perceptron

- a single layer of these devices could not do much outside of recognize some shapes
- as such, community gave up on neural nets for some time

1986 Parallel Distributed Processing (PDP) was released that described improving NNs by adding more layers. In addition to the following architecture:

1. A set of *processing units*
2. A *state of activation*
3. An *output function* for each unit
4. A *pattern of connectivity* among units
5. A *propagation rule* for propagating patterns of activities through the network of connectivity
6. An *activation rule* for combining the inputs impinging on a unit with the current state of that unit to produce an output for the unit
7. A *learning rule* whereby patterns of connectivity are modified by experience
8. An *environment* within which the system must operate

Love the “Teach the whole game” education methodology!

- if you’re teaching basketball to a young child, you start by watching the NCAA tournament with them. Not by taking them through ball handling drills and doing defensive slides!
- I would argue after that the next best step is to start by actually playing the game! Go play pick up with a group of random people you don’t know and see how much fun it is.
- Harvard Professor David Perkins argues the same thing applies to all subjects, not just sports.

*universal approximation theorem* shows that this function can solve any problem to any level of accuracy

Classification vs Regression

A classification model is one which attempts to predict a class, or category.

A regression model is one which attempts to predict one or more numeric quantities, such as a temperature or a location.

fastai will *always* show you your model's accuracy using *only* the validation set, *never* the training set. This is absolutely critical, because if you train a large enough model for a long enough time, it will eventually memorize the label of every item in your dataset!

Validation Set: When you train a model, you must *always* have both a training set and a validation set, and must measure the accuracy of your model only on the validation set. If you train for too long, with not enough data, you will see the accuracy of your model start to get worse; this is called *overfitting*. fastai defaults `valid_pct` to `0.2`, so even if you forget, fastai will create a validation set for you!

The concept of a metric may remind you of *loss*, but there is an important distinction. The entire purpose of loss is to define a "measure of performance" that the training system can use to update weights automatically. In other words, a good choice for loss is a choice that is easy for stochastic gradient descent to use. But a metric is defined for human consumption, so a good metric is one that is easy for you to understand, and that hews as closely as possible to what you want the model to do.

Using a pretrained model for a task different to what it was originally trained for is known as *transfer learning*.

Fine-tuning: A transfer learning technique where the parameters of a pretrained model are updated by training for additional epochs using a different task to that used for pretraining.

*Machine learning* is a discipline where we define a program not by writing it entirely ourselves, but by learning from data. *Deep learning* is a specialty within machine learning that uses *neural networks* with multiple *layers*

Tabular: Data that is in the form of a table, such as from a spreadsheet, database, or CSV file. A tabular model is a model that tries to predict one column of a table based on information in other columns of the table.