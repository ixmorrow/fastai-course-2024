# Lecture 2: Deployment

[Link to Chapter from Fast.ai Book](https://colab.research.google.com/github/fastai/fastbook/blob/master/02_production.ipynb)

[Link to Lesson](https://course.fast.ai/Lessons/lesson2.html)

[Link to Lecture on YouTube](https://youtu.be/F4tvM4Vb3A0)

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