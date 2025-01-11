## Lecture 7: Collaborative Filtering

Collaborative filtering is a technique used in recommendation systems to predict a user's preferences based on the preferences of other users or items. 

* looks at what products the current user has used or liked, find other users that have used or liked similar products and then recommend other products that those users have used or liked.

Latent Factors - hidden or unobservable variables that help explain patterns in data. Not directly measurable but can be inferred from observed data.

*  reduce the dimensionality of data by summarizing complex patterns into a smaller number of meaningful components

* For Movies: genre, style, director popularity.
* For Users: genere preference, affinity for certain directors.

To find the mathematical representation of latent factors, take the dot product of two vectors that represent their interation.

* The interaction (e.g., a user's rating of a movie) is modeled as a function (e.g. dot product) of their latent factors.

Steps to collaborativ filtering with latent factors:

1. Randomly initialize some paramters representing a set of latent factors. Decide how many to user.

2. Calculate predictions. Do this by taking the dot product of each user latent factors and item latent factors. Result is the correlation of the two latent factors. A high result indicates the user's interests align with this item. A low result indicates the user's interests and this item do not align.

3. Calculate our loss.

Embedding - essentially a way to represent categorical data (like words, users, or items) as numerical vectors in a continuous space. These vectors capture relationships and similarities between the categories.

A One-Hot-Encoded matrix is a representation where each category is mapped to a unique binary vector with all zeros except for a single 1 at the position corresponding to that cateogory.

For example if we have 4 categories (cat, dog, bird, fish), the one-hot encodedings are:

```python
cat  -> [1, 0, 0, 0]
dog  -> [0, 1, 0, 0]
bird -> [0, 0, 1, 0]
fish -> [0, 0, 0, 1]
```

Embedding Matrix is a learnable matrix where each row corresponds to the vector representation of a category. Think of it like a dictionary or a translation from the category to a unique vector of floats that represent that category.

```python
[ 0.2  0.8  0.5 ]  <- Vector for "cat"
[ 0.1  0.3  0.7 ]  <- Vector for "dog"
[ 0.6  0.2  0.4 ]  <- Vector for "bird"
[ 0.9  0.5  0.1 ]  <- Vector for "fish"
```

When you multiply a one-hot encoded vector by the embedding matrix, it "selects"/returns the embedding matrix represented by the one-hot encoded vector. This is because the 0's of the one-hot encoded vector cancel out the other vector embeddings in the multiply and all that is left is the category represented with the 1 in the one-hot encoded vector.

```python
[0, 1, 0, 0] x [[0.2, 0.8, 0.5],
                [0.1, 0.3, 0.7],
                [0.6, 0.2, 0.4],
                [0.9, 0.5, 0.1]]
= [0.1, 0.3, 0.7]
```

Above, we multiplied the one-hot encoded vector representing the category "dog" by the embedding matrix. It's important to remember two things at this point:

1. the embedding matrix is the vector definition of each individual category

2. The one-hot encoded vector is another way to represent our categories besides their string formats. Strings are not helpful to neural nets, they operate on floats.

So, the one-hot encoded vector is an intermediary step in the translation of our categories from strings to a float value that can be interpreted by a neural net.

Additionally, you do not need to actually perform the matrix multiply between the one-hot encoded vector and the embedding matrix. You can simply index into it using the index of the 1 from the one-hot encoded vecotr. This saves a ton on compute and memory compared to performing the mat mul even if it involves simple ops multiplying by 0 and 1.

The fancy name "embedding" refers to this efficient way of representing categories using dense vectors, learned during training.

Analogy:
Think of the embedding matrix as a dictionary:
	•	Each row in the embedding matrix is like a “definition” (numerical vector) for a category (word, user, or item).
	•	One-hot encoding is like looking up a word in the dictionary by flipping through pages until you find it.
	•	Using the embedding shortcut (indexing) is like opening the dictionary directly to the correct page.

### Learning Embeddings

We will attribute to each of our users and each of our movies a random vector of a certain length (here, n_factor=5) and we will make those learnable parameters. 

We don't want to manually pick out the categories that we use to make up our latent factors. We can however manually determine the amount of factors we want our model to learn.

### Weight Decay

Weigh decay, or L2 Regularization, consists in adding to your loss function the sum of all the weights squared. Why? Because when we compute the gradients, it will add a contribution to them that will encourage the weights to be as small as possible.

Letting our model learn high/large params might cause it to fit all the data points in the training set with an overcomplex function that has very sharp changes, which will lead to overfitting.

* works by adding a penalty term to the loss function that grows as the weights of the model become larger.

* L2 regularization adds a penalty proportional to the squared magnitude of the model's weights
    * regularization encourages the optimizer to find smaller weights, reducing the model's complexity
    * smaller weights lead to smoother decision tree boundaries, which are less likely to overfit the training data

Think of weight decay as a way to "punish" the model for relying too much on large weights. This helps the model find solutions that are simpler and more generalizable, which aligns with the principle of Occam's Razor in machine learning: simpler models are often better.

Why do the size of the weights translate to model complexity?

I think this is slightly misused here in the fast.ai book. The size of the weights translates to the model sensitivity to the inputs and creates potential for overfitting, but I don't necessarily think that it means it is more complex.

* when weights are large, they will cause large changes in the output given small changes to the input
    * makes the model highly sensitive to noise during training
* large weights capture noise, not general patterns
    * smaller weights lead to smoother and more stable decision boundaries or predictions that are less sensitive to fluctuations in the data

By penalizing large weights, regularization encourages the model to distribute importance across multiple features instead of focusing too much on just a few features with very large weights.

Large weights also amplify gradients during backpropagation, leading to instability in training (exploding gradients problem).

#### What does * do in this code?

```python
def create_params(size):
    return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))
```

Here, the * is the unpacking operator. In this context, it unpacks the elements of the iterable `size` (e.g. tuple) so that they are passed as separate positional args to the `torch.zeros()` function.

This might be more clear or obvious if we used typing and declared the expected type of the size parameter.

This is not to be confused with the pointer dereference operator in C++! Python does not have pointers.