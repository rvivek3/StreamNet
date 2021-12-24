# StreamNet

This is a novel deep learning architecture for processing multivariate time series. It is currently being developed to perform real-time human activity in Smart
Homes. It holds internal state to capture the history of each sensor and updates the state with every incoming sensor value, additionally encoding the elapse time
between incoming values. This allows activities to be of arbitary length and sensor values to be fed in at arbitrary frequency.

It is being tested on the CASAS datasets: http://casas.wsu.edu/datasets/
