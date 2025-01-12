## Lecture 8: Convolutions (CNNs)

A convolution applies a kernel across an image. A kernel is a little matrix.\
* the kernel slides across an image (or convolves) multiplying each pixel by the values of the kernel.
* a kernel is a smaller size than the matrix it operates on
* the result of these multiplications are then added together

Convolutions are such an important and widely used op that PyTorch has it built in. It's called `F.conv2d` in fastai.

A channel is a single basic color in an image. Regular full-color images have 3 channels: red, green, blue.

PyTorch represents an image as a rank-3 tensor with dimensions `[channels, rows, cols]`

Kernels passed to `F.conv2d` need to be rank-4 tensors: `[channels_in, features_out, rows, cols]`.

Can add padding which is simply additional pixels around the outside of our image. Most commonely, pixels of zeros are added.
* with the right padding we can ensure that the output activation map is the same size as the original image, which can make things a lot simpler when we construct our architectures.

So far we have moved the kernel one pixel at a time across the grid. But we can move over two pixels after each kernel application if we wanted. This step size is called strides and this would be known as a stride-2 convolution.

The most common kernel size is 3x3 and the most common padding 1. 

As you will see, stride-2 convs are useful for decreasing the size of our outputs, and stride-1 convs are useful for adding layers without changing the output size.

Genreal formula for determining size of output activation map is:
`(n + 2*pad -ks)//stride + 1`

### Understanding the Convolution Equations

[CNNs from different viewpoints](https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c)

Kernel performs mat mul on pixel values. A convolution can be represented as a special kind of matrix multiplication. The weight matrix is just like the ones from traditional neural networks. However, this weight matrix has two special properties:
1. zeros are untrainable. This means that they'll stay zero throughout the optimization process.
2. Some of the weights are equal, and while they are trainable, they must remain equal. These are called shared weights.

Zeros in the image represent pixels the kernel cannot touch. Each row of ht eweight matrix corresponds to one application of the filter.

Instead of manually setting the values of the different kernels like we did above, we can simply "learn" these values! We already know how to do this with SGD. In effect, the model will learn the features that are useful for classification.

Deeper we go into the CNN, the larger the receptive field for an activation in that layer. A large receptive field means that a large amount of the input image is used to calculate each activation in that layer. We now know that in the deeper layers of the network we have semantically rich features, corresponding to larger receptive fields. Therefor, we'd expect that we'd need more weights for each of our features to handle this increasing complexity.

Like our hidden size that represented the numbers of neurons in a linear layer, we can decide to have as many filters as we want and each of them will be able to specialize, some to detect horizontal edges, others to detect vertical edges and so forth.

In one sliding window, we have a certain number of channels and we need to same number of filters, we don't use the same kernel for all the channels.
* on each channel, we multiply the elements of our window by the elements of the corresponding filter then sum the results and sum over all the filters.

A neural net will automatically discover features and find values in kernels that will be able to detect things like edges in an image.

1-cycle training:

Idea developed by Leslie Smith to adjust the learning rate of a training run throughout the training. Schedule has two phases: one where the learning rate grows from the minimum value to the maximum value (warmup) and one where it decreases back to the minimum value (annealing).

Allows us to use a much higher max learning rate which gives two benefits:
* by training with higher learning rates, we train faster -- super convergence
* We overfit less because we skip over the sharp local minima to end up in a smoother and therfore more generalizable part of the loss
    * second point is interesting and subtle, based on the observation that a model that generalizes well is one whose loss would not change very much if you changed the input by a small amount
    * related to the weight decay discussed in the previous lecture. lower weight magnitude leads to less variablability to noise in training data

We don't jump straight to a high learning rate. Instead we start at a low lr and we allow the optimizer to gradually find smoother and smoother ares of our parameters by gradually going to higher and higher learning rates.

Then once we have found a nice smooth area for our params, we want to find the best part of that area so we lower our lr again.

Momentum - technique where optimizer takes a step not only in direction of the gradients, but also that continues in the direction of previous steps as well.
* step takes into account previous steps
* harder to change directions, momentum will continue carrying in same direction

### Batch Normalization

Batch Normalization - works by taking an average of the mean and the standard deviations of the activations of a layer and using those to normalize the activations
* also includes two learnable params: `gamma` and `beta`

batch norm is used to improve training efficiency and performance.

Models with batchnorm layers tend to generalize better than models that don't contain them. 
* Each mini-batch will have a somewhat different mean and standard deviation than other mini-batches. Therefore, the activations will be normalized by different values each time.
* In order for the model to make accurate predictions, it will have to learn to become robust to these variations. In general, adding additional randomization to the training process often helps.