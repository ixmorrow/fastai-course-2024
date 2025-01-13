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

## Embedding Distance

On a two-dimensional map we can calculate the distance between two coordinates using the formula sqrt(x^2 + y^2). For a 50-dimensional embedding we can do the same thing, except that we add up the squares of all 50 of the coordinate distances.

* distance between two movies' embedding vectors can define similarity

The approach we have followed to collaborative filtering in this chapter is called probalistic matrix factorization (PMF). Another approach is deep learning.

At this point in the chapter, our model only performs a dot product of the user and movie latent factor matrices and then adds the biases to that result before passing it through a `sigmoid_range` function. There is no deep learning involved here, it is machine learning because the latent vectors are created as `nn.Parameters` and this tells PyTorch to treat them as model weights and to calc their gradients/build a computational graph of all ops on them to be utilized in the backward pass of propagating the error/loss to the weight values. So, it is still machine learning model.

It is not deep learning until we introduce a NeuralNet that calculates the output instead of a simple dot product.

At first glance, I had a hard time understanding how this model was training and how it was "learning" anything so I'd like to spend a sec opining on that. The class is calling `create_params` for the user factors/bias and movice factors/bias. This helper function returns a pytorch tensor wrapped in a `nn.Parameter` wrapper. Because this class is inheriting from `nn.Module` pytorch knows it's a model and will look for all `nn.Parameter` items in the class. So these latent factor tensors have effectively become the parameters/weights of this model.

During the forward pass of the model, it performs a dot product of the given user's latent factor vector and the movie latent factors and sums the results. Then, it adds the biases from each of the user/movie latent vectors to the dot product sum. Finally it passes the result of that through a `sigmoid_range` function which I believe is a fastai implementation. It ensures the output is between the `self.y_range` or 0-5.5 by default.


`**kwargs` in a param list means "put any additional keyword ags into a dict called `kwargs`. And `**kwargs` in an argument list means "insert all key/value pairs in the `kwargs` dict as named arguments here". 

### PyTorch Embedding

(PyTorch Embedding Docs)[https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html]

`torch.nn.Embedding` class provides a lookup table for embeddings. It maps discrete indices (like IDs for users, or words) to dense vectore representations.

Takes a list or tensor of indices as input. Outputs a tensor of dense vectores (AKA Embeddings) corresponding to those indices.
* The embedding matrix is a learnable parameter, updated during training.

```python
import torch.nn as nn

embedding = nn.Embedding(num_embeddings, embedding_dim)
```

`num_embeddings` --> the number of unique indices in the input (total num of users, or words)

`embedding_dim` --> the size of the embedding vector for each index

Internally, it creates a matrix of shape (`num_embeddings`, `embedding_dim`) where each row corresponds to the embedding for a specific index.

To retrieve embeddings, you pass indices to the embedding and it returns the corresponding to those indices.

By default, the embedding weights are initialized using a uniform distribution.

Key Takeaways

	•	The nn.Embedding class is a lookup table for mappings from discrete indices to dense, trainable vectors.
	•	It is highly efficient compared to alternatives like one-hot encoding + matrix multiplication.
	•	Embeddings are essential for tasks involving discrete inputs, such as recommendation systems and NLP.

### Conceptuall What is a Collaborative Filtering model doing?
Goal is to predict a numerical value (rating) that a user would give to an item (movie, product) based on learned embeddings. 

Input to the model is a batch of user-item pairs. Each pair identifies: a user ID, and an Item ID. 

The model will predict a single value (rating b/t 0 and 5.5 in our case) for each user-item pair. This value represents the model's prediction of the user's preference for the item.

During training, the model learns to assing meaningful embeddings to each user and item such that:
* users with similar preferences have similar embeddings
* items with similar characteristics have similar embeddings

For a give user-item pair, the embeddings capture the "latent" features of both the user and the item.

Our model was trained on a dataset that contained the following:
* rows with a User ID, movie title, and the rating that the user gave the movie

The Rating was the y/target value and the model used the user and movie to predict it.

What the model learns:
1. User Embeddings (user_factors)
* these represent the preferences of each user in a latent feature space

2. Item Embeddings (item_factors)
* These represent the attributes of each item in the same latent feature space

3. Neural Network Weights
* The fully connected layers learn to combine user and item embeddings in a way that maps their interaction to the predicted rating

### What is Latent Space

A latent space is a representation space where complex, high-dimensional data is encoded into a lower-dimensional space that captures the essential characteristics or features of the data. The word “latent” means “hidden,” so a latent space represents hidden features or patterns that are not directly observable but are inferred from the data.

In machine learning models (like neural networks or embeddings):
	•	Inputs (like user/item IDs, images, or words) are transformed into vectors in the latent space.
	•	These vectors encode meaningful relationships or properties of the inputs in a way that facilitates the model’s tasks, like prediction or classification.


## Lecture 7 Video Notes

K-fold cross validation - split data into equal 5 sets. Train on k-1 sets of data, use the remaining 1 set for validation set. Then repeat and rotate through all k folds so that each one is used as a validation set once. Accumulate the eval metrics from each fold and average them.
* uses entire training set for both training and validation unlike a normal train/val split

Goal: Want to try to create the latent factor vectors for each user and movie through training. At beginning, we don't know what the value/score each user/movie should have, that needs to be learned.

Latent Factors -> "I don't know what things about movies matter to people, but there is probably some things. And let's just try using SGD to find them." - Jeremy Howard

Embedding is just looking something up in an array.
* You can think of an embedding as being a computational shortcut to multiplying by a one-hot-encoded vector.

Number of columns in the embedding matrix is whatever we want, however many factors we want to create!

Used CosineSimilarity in notebook to calculate the distance in latent space between two latent factors. Cosine similarity is very similra to Euclidean distance, but it also normalizes the result.

## Questions

1. What problem does collaborative filtering solve?
* Collaborative filtering gives you a way to programmatically learn similarities between abstract things like movies, users, etc. With collaborative filtering, you can train a model to use these similarities to predict items and/or users based on what the model has learned about a given user/element. For example, it can be used to predict what movies a user might like based on how the types of movies they have liked and/or watched in the past. It can also determine similar users to userA and suggest content that userA liked to others that it deems as similar.
* The goal of Collaborative Filtering is to predict a numerical value (rating) that a user would give to an item based on learned embeddings.

2. How does it solve it?
* The main way it solves this is with latent factors and embeddings. In collaborative filtering, we create essentially a vector/tensor of a length of our choosing initialized with some random values. There should be a vector of equal length for every element we care about in the dataset (e.g. vec for each user and movie in catalog). These vectors are called latent factors and will represent learned characteristics of each element. These are also called embeddings as they are numerical representations of more abstract features of the data. Sticking with the user/movie example some of these could be genre, director, time period, actors, etc. We initialize these vectors as model parameters and train our model on them.
* A collaborative filtering model should treat these embeddings as model weights. It will take a series of user and item pairs as input and attempt to use its saved embeddings to predict the rating that user gave that item. It then calculates a loss/error based on how far off its prediction was from that users actual rating.
* These embeddings are learned throughout the training process and the error is backpropagated to them and their values are updated during training.
* After a model has been trained, it will be able to accurately calculate a rating a given user might give to a given item.
* Additionally, the learned embedding matrix of the user/item embeddings can be useful outside of the model. These are essentially numerical representations of the user preferences and the features of items. These can be used to calculate similarities between users and items.

3. Why might a collaborative filtering predictive model fail to be a very useful recommendation system?

4. What does a crosstab representation of collaborative filtering data look like?

5. Write the code to create a crosstab representation of the MovieLens data (you might need to do some web searching!).

6. What is a latent factor? Why is it "latent"?
* A latent factor is a lowerdimensional representation of higher-dimensional data. "Latent" means hidden, so a latent factor/space represents hidden features or patterns that are not directly observable but are inferred from the data.

7. What is a dot product? Calculate a dot product manually using pure Python with lists.
* A dot product is the elementwise multiplication of two vectors and the sum of each of those results.
```python
x = [2, 5, 7]
y = [3, 9, 4]
res = 0
for i, x in enumerate(x):
    res += x * y[i]
res
```
res = 79

8. What does pandas.DataFrame.merge do?
* Merges two DataFrames with a database-style join. The join is done of columns or indexes.

9. What is an embedding matrix?
* An embedding matrix is a tensor containing numerical representations of abstract items. For example, in this chapter we worked on movies and created matrices of lenght 50 to represent different characteristics of movies. Each of these features was a numerical value that was learned by our model. We were then able to utilize these numerical representations.

10. What is the relationship between an embedding and a matrix of one-hot-encoded vectors?
* The relationship is that performing a dot product between a one-hot-encoded vector and an embedding matrix will essentially select or index the embedding of the item that is encoded in the vector.
* How? A one-hot-encoded vector is a vector where all elements are 0's except for a single one that is 1. The index of the 1 represents the category from a list of categorical classes. The 0's in the other indices cancel out the embeddings in the embedding matrix, except for the singular 1 in the one-hot-encoded vector. The 1 * embedding doesn't alter the embedding values in any way, and so the only thing returned from this operation is the embedding from that same index. This is why you can say the dot product between the two is a way to select or index into an embedding matrix.

11. Why do we need Embedding if we could use one-hot-encoded vectors for the same thing?
* The one-hot-encoded vector is a way to numerically represent a class, but it is still represented in the abstract. A neural net cannot learn features of this class and learn to identify it based on those features. Because of this, we need to essentially translate this to a vector of numbers that each represent unique features for all classes we care about. This way the model can learn which features apply to which classes etc.

12. What does an embedding contain before we start training (assuming we're not using a pretained model)?
* Random values.

13. **(TODO)** Create a class (without peeking, if possible!) and use it.

14. What does x[:,0] return?
* It selects all rows (:) from the oth column of the tensor or array x. Essentially extracts the values form the first column for every row.

15. **(TODO)** Rewrite the DotProduct class (without peeking, if possible!) and train a model with it.

16. What is a good loss function to use for MovieLens? Why?

17. What would happen if we used cross-entropy loss with MovieLens? How would we need to change the model?

18. What is the use of bias in a dot product model?
* The bias is used as a way to say whether most people like an item or not.

19. What is another name for weight decay?
* L2 Regularization.

20. Write the equation for weight decay (without peeking!).

21. **(TODO)** Write the equation for the gradient of weight decay. Why does it help reduce weights?

22. Why does reducing weights lead to better generalization?
* Larger weights mean the model is more prone to very large swings in calculations give a small change in input. This means the model will pick up and update params based noise very easily. Smaller weights lead to smoother and more stable decision boundaries or predictions that are less sensitive to fluctuations in the data.

23. What does `argsort` do in PyTorch?
* Returns a tensor containing the indices that would sort the elments of `x` in ascending order.

24. Does sorting the movie biases give the same result as averaging overall movie ratings by movie? Why/why not?

25. How do you print the names and details of the layers in a model?
* `learn.model`

26. What is the "bootstrapping problem" in collaborative filtering?
* Collaborative filtering is heavily dependent on having a large plethora of data to pull from and train on. What if you are just getting started and don't have any data on a user's preferences, what do you recommend them? There is no one right answer to this. Probably depends on what data you do have available. Also is the reason why apps tend to ask you questions about your likes/dislikes when joining up for the first time (e.g. Spotify, Youtube, Pandora, etc.).

27. How could you deal with the bootstrapping problem for new users? For new movies?
* Ask questions of the user what they like and then use that to calculate a starting embedding representing their answers. You can then use this to recommend items that align with that initial embedding calc.
* For new movies, you can either use available data on the movie and reviews from the internet to build a starter embedding of it and then begin to recommend it to users who align with it. Then you'll want to begin updating that embedding based on your users input.

28. **(TODO)** How can feedback loops impact collaborative filtering systems?

29. **(TODO)** When using a neural network in collaborative filtering, why can we have different numbers of factors for movies and users?

30. Why is there an `nn.Sequential` in the `CollabNN` model?
* this is the PyTorch way to sequentially build a NeuralNetwork layer by layer. It can be stored in a variable and called like any other PyTorch model that inherits the `torhc.nn.Module` class.

31. **(TODO)** What kind of model should we use if we want to add metadata about users and items, or information such as date and time, to a collaborative filtering model?