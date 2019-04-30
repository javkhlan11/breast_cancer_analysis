# breast_cancer_analysis 

K-nearest-neighbors algorithm is one of the basic and handy tool for classification problem in supervised machine learning.
In order to illustrate how many neighbors gives a reliable prediction, the ratio of neighbors to accuracies are plotted. 

The original data imported from sklearn.dataset -> breast_cancer.

From the plot we can see that with only one neighbor is the accuracy perfect. However, it sinks with increasing neighbors, which leads us to bad result. 
The ideal neighbors is here at 6 which gives us testing accuracy of 94%. 
