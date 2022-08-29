# StreamNet

![Model Architecture](https://github.com/rvivek3/StreamNet/blob/master/architecture.png)

This is a novel deep learning architecture for processing multivariate time series, inspired by Temporal Graph Networks (https://github.com/twitter-research/tgn). Much of the code is original or has been heavily changed from their repo, though portions (e.g. Focal Loss, Temporal Attention layer) remains the same. This architecture was designed for human activity recognition in smart home environments.

The central idea is to hold an internal state (memory) for each sensor in the household, which captures its history and updates with every incoming sensor value, additionally encoding the elapsed time between incoming values. An attention mechanism then operates over all sensor memories to update an internal state representing the house as a whole. An MLP is then used to classify the current activity in the household at every timestep using the house state.

The intended advantages of this architecture are:

1. No activity segmentation or sliding window is required. Activity recognition is performed at any desired frequency.
2. Sensors can all have arbitrary frequencies. No synchronization is required.
3. The attention mechanism can provide a certain level of interpretability.

This architecture was evaluated on the CASAS datasets: http://casas.wsu.edu/datasets/. We found that it struggled to "ignore" irrelevant information from the past. I ultimately moved away from this idea in order to explore architectures for semantic abstraction and explanation of human activities.
