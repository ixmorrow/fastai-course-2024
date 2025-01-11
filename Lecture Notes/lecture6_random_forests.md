## Lecture 6: Random Forests

Decision trees are algorithms which continually split data into individual buckets based on a series of filters. When analyzing a piece of data, simply compare it against the filter for a given node and the result will direct it to the next appropriate node.

Random forests extend the idea of decision trees further. You can only have so many nodes in a traditional decision tree. If you scale that up indefinitely, the filters will become so specific and the data in the leaf nodes will shrink drastically. This limits the generalization of the tree as well. 

This is also known as an ensemble learning method. This whole concept reminds me of the Efficient Market Hypothesis from finance. The efficient market states that the collective actions of numerous buyers/sellers in the market ensures each operating independently based on their interpretation of the available information ensure asset prices represent the "fair value" of those assets at any given time. Ensemble/bagging is similar in that numerous ML models converge to a "correct" answer to a problem based on each model's weights and training data. In markets, this is referred to as the "wisdom of the crowd".

The idea with Random Forests is to create lots of bigger trees, and take the average of their predictions. This is known as bagging. We want each model's predictions in the averaged ensemble to be uncorrelated with each other model. That way, if we average the predictions, the average will be equal to the true target value -- that's because the average of lots of uncorrelated random errors is zero.

One way to create a bunch of uncorrelated models is to train each of them on a different random subset of the data.

Only train on a random subset of data in each tree. Say random 75% of training data each tree. Each tree will re-calculate the 75% random split to ensure they each are trained on different data.

Out-of-Bag Error (OOB) - Way of measuring prediction error on the training set by only including the calculation of a row's error trees where that row was NOT included in training. This allows us to see whether the model is overfitting, without needing a separate validation set.