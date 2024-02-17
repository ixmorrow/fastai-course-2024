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

### Chapter 4 Notes

*rank* - the number of axes or dimensions in a pytorch tensor

*shape* - the size of each axis of a tensor

The book breaks down images into their pixel values with each image as 28x28 pixels.

They then calculate the average value for each pixel in all of the 3’s and all of the 7’s respectively. Once the average is calculated, you can display this average as an image to see what the average 3 or 7 looks like in this dataset.

**Stop and Think!:** How would you calculate how similar a particular image is to each of our ideal digits? Remember to step away from this book and jot down some ideas before you move on! Research shows that recall and understanding improves dramatically when you are engaged with the learning process by solving problems, experimenting, and trying new ideas yourself

- To compare a particular image to the “ideal” or average 3/7 you can iterate over each pixel in both images and compare how different they are
- if the images are similar then the differences pixel by pixel will be pretty minimal
- use Root Mean Squared Error or Mean Absolute Difference

*NumPy Array* - A NumPy array is a multidimensional table of data, with all items of the same type. Since that can be any type at all, they can even be arrays of arrays, with the innermost arrays potentially being different sizes—this is called a "jagged array.”

*PyTorch Tensor* - A PyTorch tensor is nearly the same thing as a NumPy array, but with an additional restriction that it has to use a single basic numeric type for all components. Cannot be jagged.

- PyTorch tensors come with GPU support and their methods are able to execute on GPUs much faster than a traditional CPU

```python
# how to create a numpy array or pytorch tensor
data = [[1,2,3],[4,5,6]]
arr = array (data)
tns = tensor(data)
```

To measure model performance we will use a metric called *accuracy.* We will derive this metric over a *validation* set. This is so that we don’t inadvertently overfit - that is, train a model to work well only on our training data.

The MNIST dataset used in the example in this chapter already has a separate dataset for validation.

- this is what we’ll use to calculate a metric measuring the quality of our first try model, which measures distance from an ideal image

Broadcasting - when PyTorch tries to perform a simple operation between two tensors of different ranks, it will automatically expand the tensor with the smaller rank to have the same size as the one with the larger rank. Broadcasting is an important capability that makes tensor code much easier to write.

Important:

- PyTorch doesn't *actually* copy `mean3` 1,010 times. It *pretends* it were a tensor of that shape, but doesn't actually allocate any additional memory
- It does the whole calculation in C (or, if you're using a GPU, in CUDA, the equivalent of C on the GPU), tens of thousands of times faster than pure Python (up to millions of times faster on a GPU!).

This is true of all broadcasting and elementwise operations and functions done in PyTorch. *It's the most important technique for you to know to create efficient PyTorch code.*

Instead of calculating how different each pixel in an example is vs the average pixel values, we can create a function that assigns a weight to each pixel. E.g. pixels in the lower right of an image of a 7 will not be activated majority of the time. Hence, these pixels should have low weight values. With a formula we can predict the likelihood an image is a 7 based on the value of it’s pixels and the weighted formula.

1. I*nitialize* the weights.
2. For each image, use these weights to *predict* whether it appears to be a 3 or a 7.
3. Based on these predictions, calculate how good the model is (its *loss*).
4. Calculate the *gradient*, which measures for each weight, how changing that weight would change the loss
5. *Step* (that is, change) all the weights based on that calculation.
6. Go back to the step 2, and *repeat* the process.
7. Iterate until you decide to *stop* the training process (for instance, because the model is good enough or you don't want to wait any longer).

These 7 steps are the key to the training of all deep learning models.

- Initialize:: We initialize the parameters to random values. This may sound surprising. There are certainly other choices we could make, such as initializing them to the percentage of times that pixel is activated for that category—but since we already know that we have a routine to improve these weights, it turns out that just starting with random weights works perfectly well.
- Loss:: This is what Samuel referred to when he spoke of *testing the effectiveness of any current weight assignment in terms of actual performance*. We need some function that will return a number that is small if the performance of the model is good (the standard approach is to treat a small loss as good, and a large loss as bad, although this is just a convention).
- Step:: A simple way to figure out whether a weight should be increased a bit, or decreased a bit, would be just to try it: increase the weight by a small amount, and see if the loss goes up or down. Once you find the correct direction, you could then change that amount by a bit more, and a bit less, until you find an amount that works well. However, this is slow! As we will see, the magic of calculus allows us to directly figure out in which direction, and by roughly how much, to change each weight, without having to try all these small changes. The way to do this is by calculating *gradients*. This is just a performance optimization, we would get exactly the same results by using the slower manual process as well.
- Stop:: Once we've decided how many epochs to train the model for (a few suggestions for this were given in the earlier list), we apply that decision. This is where that decision is applied. For our digit classifier, we would keep training until the accuracy of the model started getting worse, or we ran out of time.

Gradients are used as a performance optimization to quickly calculate whether our loss will go up or down when we adjust our parameters up or down. In other words, the gradients will tell us how much we have to change each weight to make our model better.

```python
xt = tensor(3.).requires_grad_()
```

`requires_grad_` is a method from PyTorch that we use to tell PyTorch we want to calculate gradients with respect to that variable at that value. It is essentially tagging the variable, so PyTorch will remember to keep track of how to compute gradients of the other, direct calculations on it that you will ask for.

The gradients only tell us the slope of our function, they don’t actually tell us exactly how far to adjust the params. But it does give us some idea of how far, if the slope is very large then that may suggest that we have more adjustments to do, whereas if it’s small that may suggest we are close to the optimal value.

Deciding how to change our params based on the gradients is an important part of the deep learning process. Nearly all approaches start by multiplying the gradient by some small number, called the *learning rate*.

Most learning rates are between 0.001 and 0.1 - once you’ve picked one you can adjust your params using this simple function:

```python
w -= gradient(w) * lr
```

Notice how we subtract the `gradient * lr` from the param to update it. This allows us to adjust the param in the direction of the slope by increasing it when the slope is negative and decreasing it when the slope is positive. We want to adjust our params in the direction of the slope because our goal is to minimize the loss.

Parameters: The *weights* and *biases* of a model. The weights are the `w` in the equation `w*x+b`, and the biases are the `b` in that equation.

```python
def mnist_loss(predictions, targets):
    return torch.where(targets==1, 1-predictions, predictions).mean()
```

We're using a new function, `torch.where(a,b,c)`. This is the same as running the list comprehension `[b[i] if a[i] else c[i] for i in range(len(a))]`, except it works on tensors, at C/CUDA speed. In plain English, this function will measure how distant each prediction is from 1 if it should be 1, and how distant it is from 0 if it should be 0, and then it will take the mean of all those distances.

> note: Read the Docs: It's important to learn about PyTorch functions like this, because looping over tensors in Python performs at Python speed, not C/CUDA speed! Try running help(torch.where) now to read the docs for this function, or, better still, look it up on the PyTorch documentation site.
> 

Sigmoid function can guarantee all values are between 0 and 1

```python
def sigmoid(x): return 1/(1+torch.exp(-x))
```

The key difference is that the metric is to drive human understanding and the loss is to drive automated learning. To drive automated learning, the loss must be a function that has a meaningful derivative. It can't have big flat sections and large jumps, but instead must be reasonably smooth. This is why we designed a loss function that would respond to small changes in confidence level. This requirement means that sometimes it does not really reflect exactly what we are trying to achieve, but is rather a compromise between our real goal and a function that can be optimized using its gradient. The loss function is calculated for each item in our dataset, and then at the end of an epoch the loss values are all averaged and the overall mean is reported for the epoch.

Metrics, on the other hand, are the numbers that we really care about. These are the values that are printed at the end of each epoch that tell us how our model is really doing. It is important that we learn to focus on these metrics, rather than the loss, when judging the performance of a model.

Optimization step - change or update the weights base don the gradients. We could iterate over the entire dataset, calculate the loss/gradient and take the average or we can calculate it and take a step on each item. Neither of these is ideal. Calculating it for the whole dataset would take a very long time. Calculating it for a single item would not use much information, so it would result in a very imprecise and unstable gradient. That is, you'd be going to the trouble of updating the weights, but taking into account only how that would improve the model's performance on that single item.

So instead we calc the average loss for a few data items at a time. This is called a mini-batch. Number of items in your mini-batch is called your batch size.

As we saw in our discussion of data augmentation in <>, we get better generalization if we can vary things during training. One simple and effective thing we can vary is what data items we put in each mini-batch. Rather than simply enumerating our dataset in order for every epoch, instead what we normally do is randomly shuffle it on every epoch, before we create mini-batches. PyTorch and fastai provide a class that will do the shuffling and mini-batch collation for you, called `DataLoader`.

A `DataLoader` can take any Python collection and turn it into an iterator over mini-batches, like so:

```python
coll = range(15)
dl = DataLoader(coll, batch_size=5, shuffle=True)
list(dl)
```

note: In place Operations: Methods in PyTorch whose names end in an underscore modify their objects *in place*. For instance, `bias.zero_()` sets all elements of the tensor `bias` to 0.

Rectified Linear Unit (ReLU) - replace very negative number with a zero, also known as an activation function

At this point we have something that is rather magical:

1. A function that can solve any problem to any level of accuracy (the neural network) given the correct set of parameters
2. A way to find the best set of parameters for any function (stochastic gradient descent)

`|Forward pass | Applying the model to some input and computing the predictions.`

`|Gradient | The derivative of the loss with respect to some parameter of the model.`

`|Backward pass | Computing the gradients of the loss with respect to all model parameters.`

`|Gradient descent | Taking a step in the directions opposite to the gradients to make the model parameters a little bit better.` (updating the parameters)

`|Learning rate | The size of the step we take when applying SGD to update the parameters of the model.`