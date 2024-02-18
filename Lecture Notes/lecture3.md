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

Python list comprehension to create a plain list of the single image tensors

```python
seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
len(three_tensors),len(seven_tensors)
```

note: List Comprehensions: List and dictionary comprehensions are a wonderful feature of Python. Many Python programmers use them every day, including the authors of this book—they are part of "idiomatic Python." But programmers coming from other languages may have never seen them before. There are a lot of great tutorials just a web search away, so we won't spend a long time discussing them now. Here is a quick explanation and example to get you started. A list comprehension looks like this: `new_list = [f(o) for o in a_list if o>0]`. This will return every element of `a_list` that is greater than 0, after passing it to the function `f`. There are three parts here: the collection you are iterating over (`a_list`), an optional filter (`if o>0`), and something to do to each element (`f(o)`). It's not only shorter to write but way faster than the alternative ways of creating the same list with a loop.

3 Dimensional Tensor == rank 3 tensor

To stack individual tensors in a collection into a single tensor, use the PyTorch function `stack`

```python
stacked_sevens = torch.stack(seven_tensors).float()/255
stacked_threes = torch.stack(three_tensors).float()/255
stacked_threes.shape
```

```bash
torch.Size([6131, 28, 28])
```

This tells us the shape of the tensor. It means that there are 6131 images in this tensor, each with 28 by 28 pixels. You have to have the context of what the tensor is representing in order to fully comprehend its shape. Nothing about the shape tells us that the first axis is the number of images, etc.

Length of a tensor’s shape is its rank. Otherwise known as the number of axes or dimensions in a tensor. 

Shape is the size of each axis of a tensor.

Can easily calculate the mean over every single tensor like so

```bash
mean3 = stacked_threes.mean(0)
show_image(mean3);

mean7 = stacked_sevens.mean(0)
show_image(mean7);
```

How do you calculate the difference of a specific example of a 3 or 7 versus or ideal average 3/7?

- Take the mean of the *absolute value* of differences (absolute value is the function that replaces negative values with positive values). This is called the *mean absolute difference* or *L1 norm*
- Take the mean of the *square* of differences (which makes everything positive) and then take the *square root* (which undoes the squaring). This is called the *root mean squared error* (RMSE) or *L2 norm*.

PyTorch already provides both of these as *loss functions*. You'll find these inside `torch.nn.functional`, which the PyTorch team recommends importing as `F` (and is available by default under that name in fastai):

```bash
F.l1_loss(a_3.float(),mean7), F.mse_loss(a_3,mean7).sqrt()
```

Here `mse` stands for mean squared error and `l1` refers to the standard mathematical jargon for mean absolute value (L1 norm)

NumPy arrays are very similar to PyTorch tensors, however they do not support using the GPU or calculating gradients, which are both critical for deep learning.

Python is slow compared to many languages. Anything fast in Python, NumPy, or PyTorch is likely to be a wrapper for a compiled object written (and optimized) in another language—specifically C. In fact, **NumPy arrays and PyTorch tensors can finish computations many thousands of times faster than using pure Python.**

A numpy array is a multidimensional table of data, with all items of the same type. Since that can be any type at all, they can even be arrays of arrays, with the innermost arrays potentially being different sizes—this is called a "jagged array.” By "multidimensional table" we mean, for instance, a list (dimension of one), a table or matrix (dimension of two), a "table of tables" or "cube" (dimension of three), and so forth. 

- If the items are all of some simple type such as integer or float, then NumPy will store them as a compact C data structure in memory. This is where NumPy shines. NumPy has a wide variety of operators and methods that can run computations on these compact structures at the same speed as optimized C, because they are written in optimized C.

A PyTorch tensor is nearly the same thing as a NumPy array but with an additional restriction that unlocks some additional capabilities. It is also a multidimensional table of data, with all items of the same type. However, the restriction is that a tensor cannot use just any old type—it has to use a single basic numeric type for all components. For example, a PyTorch tensor cannot be jagged. It is always a regularly shaped multidimensional rectangular structure.

- one additional capability of a PyTorch tensor is that these structures can live on the GPU in which case their computation will be optimized for the GPU and can run much faster.
- PyTorch can automatically calculate derivatives of these operations, including combinations of operations.

### Computing Metrics Using Broadcasting

A metric is a number that is calc’d based on the predictions of our model in order to tell us how good our model is.

This should be calculated over a validation set, not the training set. Reason being is that we don’t want our model to overfit and train to work well only on our training data.

In order to calculate a metric for overall accuracy, we need to calculate the distance to the ideal 3 for every image in the validation set. How do we do that calculation?

The magic trick is that PyTorch, when it tries to perform a simple subtraction operation between two tensors of different ranks, will use *broadcasting*.

- It will automatically expand the tensor with the smaller rank tohav ethe same size as the one with the larger rank.
- Broadcasting is an important capability that makes tensor code much easier to write

For example:

```bash
tensor([1,2,3]) + tensor(1)

tensor([2, 3, 4])
```

So in this case, PyTorch treats `mean3`, a rank-2 tensor representing a single image, as if it were 1,010 copies of the same image, and then subtracts each of those copies from each 3 in our validation set.

There are a couple of important points about how broadcasting is implemented, which make it valuable not jsut for expressivity by also for performance:

- PyTorch doesn’t actually copy mean3 1010 times. It pretends it were a tensor of that shape, but doesnt actually allocate any additional memory
- It doe sthe whole calculation in C (or if you’re using a GPU, in CUDA (only on NVDA GPU?)), tens of thousands of times faster than pure Python (up to millions of times faster on a GPU)

Change weight each step in SGD in the direction of the slope!

Backpropagation - name given to the process of calculating the derivatives of each layer.

Updating a parameter in SGD

```bash
w -= gradient(w) * lr
```

Notice how we subract the `gradient * lr` from the param to update. This allows us to adjust the param in the direction of the slope by increasing the param when the slope is negative and decreasing the param when the slope is positive. We want to adjust our params in the direction of the slope beause our goal in deep learning is to minimize the loss.

### Questions

1. How is a grayscale image represented on a computer? How about a color image?
    1. Both represented as pixels with an integer value
2. How are the files and folders in the `MNIST_SAMPLE` dataset structured? Why?
    1. They are structured in a way that the data is split into training and validation data. This is done ahead of time for you. Reason being is that when training NN you should always use two sets of data. One is to train and the validation is to test your trained model on new data that it hasn’t seen during training. You use the validation set to measure the model’s performance. This is to avoid over fitting and allowing a model to fit its parameters to a very specific dataset and then test it on that same dataset.
3. Explain how the “pixel similarity” approach to classifying digits works.
    1. The idea is that you iterate over every single pixel for each image of a 3 and every single pixel for each image of a 7. You calculate an average value for each pixel for all 3’s and all 7’s. This represents the “average” or “ideal” 3 or 7. Next, you can predict the likelihood that an image is a 3 or 7 by comparing it to these average 3’s and 7’s. You can compare these images pixel by pixel, and the answer with the higher probability is the final prediciton.
4. What is a list comprehenstion? Create one now that selects odd numbers from a list and doubles them.
    1. A list comprehension is a short hand way of writing a for loop that can iterate over a list of data in python. With list comprehension you can specify a filter so that it will only iterate over items in the dataset that meet that requirement. You can even perform an action on the filtered items.
    2. List comprehension offers a shorter syntax when you want to create a new list based on the values of an existing list.
    
    ```python
    newList = [o*2 for o in numsList if 0%2 != 0]
    ```
    
    1. What is a “rank-3 tensor”?
        1. A rank-3 tensor is a tensor with a shape size of 3, meaning the tensor has 3 dimensions.
        
    2. What is the difference between tensor rank and shape? How do you get the rank from the shape?
        1. The rank is the size of the shape, or the number of dimensions in the tensor. The shape is the size of each individual dimension.
    3. What are RMSE and L1 norm?
        1. RMSE - Root Mean Squared Error, mean of the square of differences and then take the square root
        2. L1 norm - the mean of the absolute value of differences
    4. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?
        1. By utilizing PyTorch tensors or NumPy arrays. These data types make use of optimized C under the hood, and PyTorch tensors execute on GPUs if there are available which is thousands of times faster than true python.
    5. Create a 3x3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.
    
    ```python
    data = [[1,2,3],[4,5,6], [7,8,9]]
    tns = tensor(data)
    tns = tns*2
    tns[2:]
    tns
    ```
    
    1. What is broadcasting?
        1. Broadcasting is the ability to automatically convert a tensor that is of a different shape to match the shape of another tensor when performing operations on the two.
        2. From my notes: Broadcasting - when PyTorch tries to perform a simple operation between two tensors of different ranks, it will automatically expand the tensor with the smaller rank to have the same size as the one with the larger rank. Broadcasting is an important capability that makes tensor code much easier to write.
    2. Are metrics generally calculated using the training set, or the validation set? Why?
        1. Metrics are calculated using the validation set. This is to prevent a model from over fitting and optimizing its parameters to just the training data. Use the validation set as a way to measure the model’s performance by feeding it data that it has not seen before in training.
    3. What is SGD?
        1. SGD is Stochastic Gradient Descent. It is an algorithm for optimizing a given function?
    4. Why does SGD use mini-batches?
        1. Part of SGD is evaluating the current algorithm’s parameters, and modifying them by “stepping” them in a direction that improves the model. This process could be done on every single prediction the model gives during training, or you can do it all at once at the end of training. This is where mini-batches come in, instead of doing either of those you can break the training data into mini-batches and perform this evaluation and parameter update at the end of each batch.
    5. what are the seven steps in SGD for machine learning?
        1. I*nitialize* the weights.
        2. For each image, use these weights to *predict* whether it appears to be a 3 or a 7.
        3. Based on these predictions, calculate how good the model is (its *loss*).
        4. Calculate the *gradient*, which measures for each weight, how changing that weight would change the loss
        5. *Step* (that is, change) all the weights based on that calculation.
        6. Go back to the step 2, and *repeat* the process.
        7. Iterate until you decide to *stop* the training process (for instance, because the model is good enough or you don't want to wait any longer).
    6. How do we initialize the weights in a model?
        1. Either by downloading a model with pre-trained weights for us to use, or generate some random values. 
    7. What is “loss”?
        1. Loss is a measure of how well the model is performing in a way that is very responsive to very small changes to the model’s parameters. The loss is used by the model itself to evaluate its performance and the goal of SGD is to step the parameters in a fashion that will minimize the loss overall.
    8. Why can’t we always use a high learning rate?
        1. The learning rate is a value that is used in the “step” part of the SGD process. Using too large of a learning rate leads to the step being too large and it is not an optimal way to change our parameters. Using too small of a learning rate would likely lead to needing a lot more training iterations to get to the optimal loss.
    9. What is a “gradient”?
        1. A gradient is a measure of how much a given point/value on a graph is changing at the moment in time. Otherwise known as the derivative.
        2. Conceptually, the gradient tells us and the model in which direction the specific parameter needs to change in order to reach a value that minimizes the loss.
    10. Do you need to know how to calculate gradients yourself?
        1. No. Fastai and PyTorch both provide apis you can use to calculate the gradients for you.
    11. Why can’t we use accuracy as a loss function?
        1. Accuracy is a good metric for humans to use, but the accuracy of a model does not change very much given a small change to its parameters. This makes it very difficult for the model to evaluate how its performing. Instead, loss is used because the loss does change quite a lot for just a small change to the parameters.
    12. Draw the sigmoid function. What is special about its shape?
        1. guarantees all values are between 0 and 1
    13. What is the difference between a loss function and a metric?
        1. Loss function is used by the model it evaluate its performance. Accuracy is used by humans.
    14. What is the function to calculate new weights using a learning rate?
        1. `w -= w *lr`
    15. what does the `DataLoader` class do?
        1. A `DataLoader` can take any Python collection and turn it into an iterator over mini-batches
        2. Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.
    16. Write pseudocode showing the basic steps taken in each epoch for SGD.
    
    ```python
    # 1. init params
    params = torch.randn(3).requires_grad_()
    
    # 2. predict using the params
    def f(x):
    	m, b = params
    	return m*x + b
    
    preds = f(train_x)
    
    # calculate the loss on predictions
    def calc_loss(predictions, targets):
    	return torch.where(targets==1, 1-predictions, predictions).mean()
    
    loss = calc_loss(preds, train_y)
    
    # calculate the gradient
    loss.backward()
    
    # step the parameters and reset gradient to 0
    for p in params:
    	p.data -= p.grad*lr
    	p.grad.zero_()
    
    # repeat process all over again
    ```
    
    1. Create a function that, if passed two arguments `[1,2,3,4]` and `'abcd'`, returns `[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]`. What is special about that output data structure?
    
    ```python
    data = [1,2,3,4]
    letters = 'abcd'
    
    def combine(x, y):
        return list(zip(x, y))
    
    combine(data, letters)
    ```
    
    Output: `[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]`
    
    1. This output is a list of tuples. A Dataset in PyTorch is required to return a tuple of (x,y) when indexed.
    
    27. What does `view` do in PyTorch?
    
    1. `view` is a PyTorch method that changes the shape of a tensor without changing its contents.
    1. what are the “bias” parameters in a neural network? Why do we need them?
    2. What does the `@` operator do in Python?
        1. Matrix multiplication
    3. What does the `backward` method do?
        1. Calling backward() on a loss calculates the gradients for each parameter in the model
    4. Why do we have to zero the gradients?
        1. Because if you do not, each subsquent time you call `loss.backward()` the newly calculated gradients will be added to the existing gradients which is not what we want. Re-setting the gradients back to 0 after each iteration ensures this doesn’t happen.
    5. What information do we have to pass to `Learner`?
        1. the DataLoaders, the model, the optimization function (which will be passed the parameters), the loss function, and optionally any metrics to print
    6. Show Python or pseudocode for the basic steps of a training loop.
        
        ```python
        def train_epoch(model):
            for xb,yb in dl:
                calc_grad(xb, yb, model)
                opt.step()
                opt.zero_grad()
        ```
        
    7. What is “ReLu”? Draw a plot of it for values from -2 to +2
        1. ReLu stands for Rectified Linear Unit, all it does is set all negative values to 0. So a plot of ReLu from -2 to +2 would look like a hockey stick
    8. What is an “activation function”?
        1. An activation function is a function that preps the output form one layer of a neural net for the next. ReLu is a common activation function.
        2. A neural network contains a number of layers. Each layer is either *linear* or *nonlinear*. We generally alternate between these two kinds of layers in a neural network. Sometimes people refer to both a linear layer and its subsequent nonlinearity together as a single layer. Yes, this is confusing. Sometimes a nonlinearity is referred to as an *activation function*.
        3. Function that returns 0 for negative numbers an doesn’t change positive numbers
    9. What’s the difference between `F.relu` and `nn.ReLU`?
        1. `F.relu` is calling the relu function on the fastai API which `nn.ReLU` is calling on the PyTorch API. Other than that, there is no difference.
    10. The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?
        1. The reason is performance. With a deeper model (that is, one with more layers) we do not need to use as many parameters; it turns out that we can use smaller matrices with more layers, and get better results than we would get with larger matrices, and few layers. That means that we can train the model more quickly, and it will take up less memory.