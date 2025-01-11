## Lecture 5: From-scratch Model

https://course.fast.ai/Lessons/lesson5.html

Looking at creating a neural network from scratch using Python and PyTorch.

Based partly on chapters 4 and 9 of the book.

Working through the [Linear Model and Neural Net from Scratch notebook](https://www.kaggle.com/code/jhoward/linear-model-and-neural-net-from-scratch])

With pytorch, a rank 2 tensor (matrix) multiplied by a rank 1 tensor (vector) broadcasts the vector to match the dimensions of the matrix. Allowing for the element-wise multiplication of the vector across each row of the matrix.

Broadcasting was invented in the APL programming language in the 60s. This acts as if there's a separate copy of the vector for every row of the matrix, so it performs ops for each row of the matrix with the vector. It does not make any copies and the whole thing is run in highly optimized C or CUDA code.

Call `requires_grad_()` on PyTorch tensor to tell PyTorch you want to calculate gradients. Need to do this to determine the direction and the magnitude that the weight value needs to change. Gradient will tell you the slope of the weight. 

* in PyTorch anything that ends with an `_` indicates an in place operation, meaning it will perform in place ops on the object you call this on

By calling `requires_grad_()` on PyTorch tensor, PyTorch will keep track of all the ops performed on that object's data so that it can calculate the gradient afterwards. Then, you can call `backward()` on the loss function and PyTorch will calc the gradients of the tensor and store them in a `grad` member variable.

```python
coeffs.requires_grad_()
loss = calc_loss(coeffs, t_indep, t_dep)
loss.backward()
coeffs.grad
```

[Why you should use a framework](https://www.kaggle.com/code/jhoward/why-you-should-use-a-framework)

[How Random Forests really work](https://www.kaggle.com/code/jhoward/how-random-forests-really-work/)

A random forest is an ensemble of trees. A tree is an ensemble of binary splits.

Binary split splits the data into two groups, hence the name "binary".
