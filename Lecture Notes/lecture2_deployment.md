# Lecture 2: Deployment

Steps to Deploying a Model in Prod:

1. Decide what problem you want to solve
2. Train a Model
3. Clean the data
4. re-train the model on the cleaned data

“Before you train the data, you train a model.” - Jeromy Howard

Data Augmentation - refers to creating random variations of our input data, such that they appear different, but do not actually change the meaning of the data.

Confusion Matrix - shows what category errors in the model making

Fast ai allows you to plot the top losses of a model. This plots the items that had the largest loss value. A prediction will have a high loss value if the model was very confident in its answer and got it wrong. Or if the model predicted the right answer, but was not very confident.

`ImageClassifierCleaner` allows you to go through and re-label training data if needed. Allows you to delete or re label training data that is labelled incorrectly.

Before you start data cleaning, always train a model to find out what things are hard to recognize in your data and to find the things that the model can help you find data problems.

CPUs when they run out of RAM will start swapping. GPUs on the other hand are done once they run out of RAM. For this reason, you must stop all notebooks once they are done running to conserve the GPUs RAM.

HuggingFace Spaces + Gradio ?

- https://huggingface.co/spaces
- https://huggingface.co/spaces/ixmorrow/minimal

pkl (”pickle”) file

*data augmentation -* synthetically generate variations of input images, such as by rotating them or changing their brightness and contrast; this also works well for text and other types of models.

From the book (circa 2020):

*combine a knowledge base of medical information with a deep learning model for generating medically correct natural language responses. This is very dangerous, because it is so easy to create content that appears to a layman to be compelling, but actually is entirely incorrect.*

- this book was released before LLMs took off, specifically two years before chat gpt was released (Nov ‘22). It seems like this statement is not really true anymore? Or maybe it could be true, as these LLMs do “Hallucinate” and produce inaccurate results sometimes. But most of the time they’re accurate!
- Just goes to show how far the space has developed over the last 4 (!!!) years

### The Drivetrain Approach

An approach to designing a ML Model from Jeromy Howard & co.

Article → https://www.oreilly.com/radar/drivetrain-approach-data-products/

Goal: Produce *actionable outcomes*

Focuses on 4 key components:

1. Defined objective - What outcome am I trying to achieve?
2. Levers - What actions can I take to achieve that outcome? What do I have control over?
3. Data - What data is available to me to execute this?
4. Models - How the levers influence the objective
    1. Only after the first 3 steps do we begin to think about predictive models. Our objective and available levers, what data we already have and what additional data we will need to collect, determine the models we can build.
    2. The models will take both the levers and any uncontrollable variables as their inputs; the outputs from the models can be combined to predict the final state for our objective.

Many AI enabled prediction services are just making predictions, rather than asking what action they want someone to take as a result of a prediction.

DataLoaders: A fastai class that stores multiple `DataLoader` objects you pass to it, normally a `train` and a `valid`, although it's possible to have as many as you like. The first two are made available as properties.

fastai’s applications all use the same basic steps and code:

- Create appropriate `[DataLoaders](https://docs.fast.ai/data.core.html#dataloaders)`
- Create a `[Learner](https://docs.fast.ai/learner.html#learner)`
- Call a *fit* method
- Make predictions or view results.

Instead of resizing or cropping images, what we normally do in practice is to randomly select part of the image, and crop to just that part. On each epoch (which is one complete pass through all of our images in the dataset) we randomly select a different part of each image. This means that our model can learn to focus on, and recognize, different features in our images. It also reflects how images work in the real world: different photos of the same thing may be framed in slightly different ways.

*Data augmentation* refers to creating random variations of our input data, such that they appear different, but do not actually change the meaning of the data.

Kaggle Notebook:

https://www.kaggle.com/ivanxm/fastai-2024-bear-classifier/edit

### Chapter 2 Questions

1. Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.
    1. The obvious answer to this is a scenario that was covered in the chapter. If our goal was to build a model that could recognize different bear types based on trail cam footage in the wild, then the model we built would not be well suited for this. For one, our model was trained on images not video and two the images it was trained on are for the most part pretty clear. Trail cam footage is not nearly as cut and dry as the images that the model was trained on. For these two reasons, the model would perform poorly on that task given the data it was trained on.
2. Where do text models currently have a major deficiency?
3. What are possible negative societal implications of text generation models?
    1. Impersonation and plagiarism. With text generation models, it is fairly easy to conjure up some text that is made to sound like someone or something it is not. In addition, you can train a model on a particular author’s writing in an attempt to generate text in their likeness. This is not necessarily plagiarism in the sense of how it’s defined today, but could be considered so in the future.
4. In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?
    1. create a buffer of required human intervention. Instead of complete automation, utilize a model to enhance/augment whatever the previously completely manual process was. 
5. What kind of tabular data is deep learning particularly good at?
    1. Any kind of tabular data
6. What's a key downside of directly using a deep learning model for recommendation systems?
    1. At the moment, it can just provide you with suggestions that people with similar interests have liked. What it cannot do, is understand the essence of an object and its content and suggest something that is similar but not well known by many others. For this reason, most of the suggestions recommendation systems provide are for things that the customer has already seen or heard of. So, they are not great at providing completely new and unique suggestions given some parameters.
7. What are the steps of the Drivetrain approach?
    1. The goal is to produce actionable outcomes. Four steps: defined objective, actions you can take to accomplish this objective, data available, what type of model to use.
8. How do the steps of the Drivetrain Approach map to a recommendation system?
    1. Objective - provide customer with new products/content that they would like and have not seen before.
    2. Actions - give customers ability to like and dislike suggestions to fine tune model
    3. Data - search history, likes, previous purchases, metadata about their demographics, customers with similar data
    4. Model - 
9. Create an image recognition model using data you curate, and deploy it on the web.
10. What are `DataLoaders`?
    1. Data structure provided by the fastai API that provides an easy way to format training data, split it into validation/training, define the modality of the model, etc.
11. What four things do we need to tell fastai to create `DataLoaders`?
    1. What type of data we are passing in (image, text, etc.) and what type of predictions to make (category, etc.)
    2. How to locate the training data. Typically a function.
    3. How to split the data between training and validation data
    4. What to use for the labels of the training data. Also a function.
12. What does the `splitter` parameter to `DataBlock` do?
    1. splits the data into training and validation data
13. How do we ensure a random split always gives the same validation set?
    1. pass in a constant seed value
14. What letters are often used to signify the independent and dependent variables?
    1. x → independent, y→ dependent
15. What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?
16. What is data augmentation? Why is it needed?
    1. Data augmentation is the process of manipulating an image in ways that transform how it appears to a model but still retaining its true meaning.
    2. Data Augmentation - refers to creating random variations of our input data, such that they appear different, but do not actually change the meaning of the data.
17. What is the difference between `item_tfms` and `batch_tfms`?
18. What is a confusion matrix?
    1. A visual graph depicting how well a model has performed during training. Can easily visualize all the predictions the model predicted correctly and incorrectly.
19. What does `export` save?
    1. a trained model and its weights in a `.pkl` file
20. What is it called when we use a model for getting predictions, instead of training?
    1. Inference
21. What are IPython widgets?
22. When might you want to use CPU for deployment? When might GPU be better?
23. What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?
    1. More latency introduced on inference if a client (edge device) needs to send all requests to a remote server and wait to receive response/answer back.
    2. Alternatively, deploying model to edge devices is not always viable depending on the size of the model and the compute capacity of the device.
24. What are three examples of problems that could occur when rolling out a bear warning system in practice?
25. What is "out-of-domain data"?
    1. Data that would be presented to a model in production but not during training
26. What is "domain shift"?
    1. the type of data that our model sees changes over time.
27. What are the three steps in the deployment process?
    1. Manual process - run model in parallel but do not solely rely on it
    2. Limited Score deployment - careful human supervision
    3. Gradual expansion