# Lecture 1: Getting Started

[Link to Chapter from Fast.ai Book](https://colab.research.google.com/github/fastai/fastbook/blob/master/01_intro.ipynb)

[Link to Lesson](https://course.fast.ai/Lessons/lesson1.html)

[Link to Lecture on YouTube](https://youtu.be/8SF_h3xF3cE)

I have just discovered this online course material from [Fast.ai](https://course.fast.ai/) and the corresponding [book](https://course.fast.ai/Resources/book.html) that goes along with it! I am not super familiar with Fast AI yet, but from what I understand it’s essentially an API built on top of PyTorch. I stumbled upon this course while looking for material to learn more about AI development and so far this looks like one of the best resources I’ve come across. The course specifically focuses on the aspect of writing code for Machine Learning and is very practical. I have decided to document my progress through this course, as I have found explaining/teaching something is the best way for to understand something at a deeper level. So, that’s what I’ll be doing in this blog series. I also have not flexed my writing muscles in some time, so this will serve as some much needed practice for me!

The Fast.ai course follows an interesting teaching method that differentiates it from most other courses. Jeromy Howard, the founder of Fast.ai and teacher of this course, follows a “Teach the whole game” methodology. He compares this teaching method to how you would teach a child a new sport. Most likely, you would start with sitting and watching a whole game with them – like an NBA game for basketball. After that, I would then try to get them exposed to what it’s like to actually play the game of basketball by going somewhere and playing in a pick up game or a rec league. Once you do that, they start to realize how fun it really is to play, but it may be apparent they are not yet very good. From here you can gradually start to mix in some sport specific training teaching the fundamentals, but being sure to get a lot of playing opportunities in the mix! This is an easy way to fall in love with the game and get instant feedback on their improvement! Well, Jeromy argues that education should be taught in a similar manner as sports. With that said, he does not start the course off teaching math, ML theory, or anything like that. He starts by showing us model implemented in code with Fast.ai. We walk through the lines of the program and Jeromy explains in depth what each one is doing. This is intentional and sets the tone for the remainder of the course.

The first lecture focused on providing an introduction to the [Fast.ai](http://Fast.ai) API and got us programming right away. We were able to build an image classification neural net in just a few lines of code! This was easily the fewest lines of code I have seen required for this. The Fast.ai API provides a lot of functionality out of the gate for you, including access to readily available training data! This was a unique feature I was not expecting. The Fast.ai API can best be described as a package that allows you to fine tune different types of models for your own use. They provide APIs that expose fully trained models for you to use and make it very, very easy to fine tune them to your own data.

Fast.ai has some new concepts that I was not familiar with beforehand. For example, the Fast.ai API has the concept of a “Learner”, which is a data structure that combines a model and the data to train the model with. Fast.ai abstracts a lot away from the developer, which is how they can provide such powerful APIs.

The first neural net was created in 1957, the Mark 1 Perceptron. A single layer of these devices could not do much outside of recognize some shapes. As such, the development community gave up on neural nets for some time. This is an astonishing fact to me. First of all, it’s remarkable that the very first neural net was created in 1957! What’s even crazier to me is how far along the technology may be if neural nets and machine learning were pursued from their inception. It makes sense that was not the case though, given how much data and compute neural nets require today. This level of technology was not even available in 1957, so there really would not have been much use for them then. That just goes to show how far ahead of its time the Mark 1 Perception was.

The next great leap in neural nets came in 1986 when Parallel Distributed Processing (PDP) was released. This described improving NNs by adding more layers. It also introduced an architecture for NNs that is pretty similar to the architecture they follow today:

1. A set of *processing units*
2. A *state of activation*
3. An *output function* for each unit
4. A *pattern of connectivity* among units
5. A *propagation rule* for propagating patterns of activities through the network of connectivity
6. An *activation rule* for combining the inputs impinging on a unit with the current state of that unit to produce an output for the unit
7. A *learning rule* whereby patterns of connectivity are modified by experience
8. An *environment* within which the system must operate

Here are some excerpts from Chapter 1 of the book that really stood out to me.

- fastai will *always* show you your model's accuracy using *only* the validation set, *never* the training set. This is absolutely critical, because if you train a large enough model for a long enough time, it will eventually memorize the label of every item in your dataset!
- Validation Set: When you train a model, you must *always* have both a training set and a validation set, and must measure the accuracy of your model only on the validation set. If you train for too long, with not enough data, you will see the accuracy of your model start to get worse; this is called *overfitting*. fastai defaults valid_pct to 2, so even if you forget, fastai will create a validation set for you!
- The concept of a metric may remind you of *loss*, but there is an important distinction. The entire purpose of loss is to define a "measure of performance" that the training system can use to update weights automatically. In other words, a good choice for loss is a choice that is easy for stochastic gradient descent to use. But a metric is defined for human consumption, so a good metric is one that is easy for you to understand, and that hews as closely as possible to what you want the model to do.
- Using a pretrained model for a task different to what it was originally trained for is known as *transfer learning*.
- Fine-tuning: A transfer learning technique where the parameters of a pretrained model are updated by training for additional epochs using a different task to that used for pretraining.
- *Machine learning* is a discipline where we define a program not by writing it entirely ourselves, but by learning from data. *Deep learning* is a specialty within machine learning that uses *neural networks* with multiple *layers*.

Well that is it for the first installment in this series. I encourage you to participate in the course yourself and/or read the book if this interests you at all! Hope this provided some insight into what the fastai course is all about!