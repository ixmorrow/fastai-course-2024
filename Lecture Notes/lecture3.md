# Lecture 3: Neural Net Foundations

How does a neural network really work?

Kaggle Notebook: https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work

Python for Data Analysis Book:

https://wesmckinney.com/book/

Loss function - how good is the model performing?

- most common is Mean Squared Error
    - distance from each data point to the curve

```python
def mse(prediction, actual):
	((prediction-actual)**2).mean()
```

- want to minimize the mean squared error

Derivative - function that tells you if you increase the input does the output increase or decrease and by how much. Rate of change of a function.

- watch first 3 videos in this [playlist for refresher on derivatives](https://www.youtube.com/playlist?list=PLybg94GvOJ9ELZEe9s2NXTKr41Yedbw7M)

Optimizers in Machine Learning are based on gradient descent

How does a neural network work?
A NN is a mathematical function that

1. multiplies each input by a number of values, these are known as parameters
2. adds them up for each group of values
3. replaces negative numbers with zeros

This is one “layer”. Then those 3 steps are repeated using the outputs of the previous layer as inputs to the next layer.

To get the function to “learn” to do something useful, we have to change the parameters to make them “better”. We do this using gradient descent.

*Tensor* - similar to an array in numpy, can be a single number, a list of numbers, a table of numbers, a table of tables of numbers, etc.

In modern neural networks, we’ll often have tens of millions of parameters to fit, or more, and thousands or millions of data points to fit them to. 

If we know the *gradient* of our mean square error function with respect to our parameters, then that means we know how adjusting a parameter will change the value of the mean squared error.

Learning rate - amount that determines how much your model adjusts a parameter by. Learning rate is a hyper parameter.

Hyper parameter - a parameter used to calculate the parameters