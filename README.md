
#Comparison of Knowledge based, Machine learning and Deep learning techniques for Handwriting Analysis 

Abstract
The process of handwritten analysis can be done by using a Knowledge based technique, machine learning techniques or  Deep learning approaches. Probabilistic graphical models help represent joint probability distribution which is a knowledge based approach and can be used in the process of handwritten comparison by training given samples of word and written by various writers and the set of characteristics for each sample. A Bayesian network model is learnt and useful inferences are performed from it. A pre-existing tool CEDAR-FOX is used to compare pairs of samples based on machine learning approach and a Siamese twin network is used for predicting the probability of the pairs of samples occurring in the same class or different class. The goal is to compare the efficiency of each of these methods in performing the analysis. 

1	Introduction

Handwriting Comparison as part of handwriting analysis is an important process in forensic AI domain. It dates back long in the history. The basis of handwriting analysis as a science is that every person in the world has a unique way of writing. It is the process of using scientific methods to determine the origins of documentation, both written and electronically produced. It helps determine if two samples are from the same or different writer. Current approaches to handwritten recognition includes usage of  many  machine learning as well as deep learning techniques. The dataset consists of cursive handwritten images of and with human observed characteristics for each image. The goal is to determine if any given pair of handwritten images as inputs are by same or different writers. In this paper, we try to compare the performance of the existing techniques using Probabilistic graphical model, a machine learning tool and a siamese twin architecture based on Convolutional neural network concept.

A probabilistic graphical model is a graph-based representation for encoding joint probability distributions over large numbers of random variables that interact with each other. They are of two types : Bayesian Networks, Markov Networks. A bayesian network is a directed graph and markov network is a undirected graph. The former is mostly used when there is a causal relationship between the random variables. In our first approach, a Bayesian network structure model as proposed in [2] for cursive writing is used which constitutes the knowledge based approach. The graph here is a probabilistic model constructed so that it may be used to quantify the degree of individualization provided by the human observed characteristics.

Simple machine learning techniques have been widely used in object recognition. Larger datasets, more powerful models and better techniques help in improving performance and preventing overfitting. In our second approach, the software system for forensic comparison of handwriting, CEDAR-FOX as proposed in [3] is used which is based on simple machine learning approach. 

Convolution neural networks have a very large learning capacity which make strong and mostly correct assumptions about the nature of images. So, our last approach consists of a deep learning model, with convolutional siamese twin architecture where two images are simultaneously propagated through a twin CNN model and later, merged and passed through a linear classifier for determining the result. Due to involvement of huge number of parameters, there is a fair chance of overfitting, which is avoided by use of a very large dataset as it consists of a number of similar and different writer image pairs.

By using the above mentioned approaches, we compare the efficiency of each of these methods in forensic analysis of handwritings. A comparison of the above techniques is necessary for understanding a better approach in the process of handwriting analysis.  

2	Bayesian Network Model
Bayesian Network(BN) is a directed acyclic graph which represents different features of the data as a set of variables and their dependencies on each other as conditional probabilities. The paper describes two BN approaches taken and reason why one BN is preferred over another. The BN models described in this paper is created for handwritten samples of AND dataset. The BNs compares features of two ‘and’ images. BayesianModel of Pgmpy library is used to create the BN. 
2.1	AND Dataset
In the dataset, writing samples of word ‘and’ have been taken from 342 writers. Each writer has 3 samples which have been taken from 3 different pages making the total number of samples as 1026. A set of 9 features describes the each sample. Description of all features and their values is provided in Table 1[2]. These features have been assigned manually by examiners. Table 2 shows sample of and dataset. The numeric part shows the writer ID and alphabet indicates page. 

2.2	Bayesian Network: First Model
First Model, the BN used is a multinet with 9 nodes for each image and one decision node, in total 19 nodes. First 9 nodes, X1 to X9, denotes feature vector of first ‘and’ image, second 9 nodes, X10 to X18, denote feature vector of second ‘and’ image and node X19 denotes the decision variable. The decision variable takes value ‘1’ when the images belong to the same writer and ‘0’ when images belong to different writers. The BN used is shown in fig. 1 is used as base model for nodes X1 to X9 and nodes X10 to X18 and decision node X19 is connected to node X2 and X11. This approach has several drawbacks: 
High data unbalance: The data belonging to same writer is considerably less compared to data belonging to different writers. This resulted in high conditional probability for 0 value of decision node.. Thus, output of all comparisons of image pairs resulted that they belonged to different writers, as predict function of pgmpy returns value with higher probability. 
Effect of Parent Nodes: The decision node is impacted only by the immediate parent nodes. Thus, change in any other node does not have impact on the decision node. This leads to incorrect predictions.
2.3	Bayesian Network: Second Model
In second model, two BNs are used and each has 9 nodes. The BN model used is show in Fig. 2, with joint probability . In an image pair, the features of one image is subtracted from the other and absolute value is fed to the network for training. Of the two BNs, one is trained on the data for same writer and other BN is trained on data for different writers, thus, resulting in different Conditional Probability Distributions(CPDs). For a test sample of two images, Joint probability, multiplication of CPDs, is calculated for both BNs and compared. If the probability of BN of similar data is higher, the test images are considered to be of the same writer else they are considered of different writers.
 Whether two images belong to same writer or not can also be determined using log likelihood ratio(LLR). LLR is logarithm of the joint probability of BN of similar data divided by joint probability of BN of dissimilar data.  If LLR is positive, images belong to same writer and if LLR is negative, images belong to different writers.
 
Figure 1: Bayesian Network used in [4]	       Figure 2: Bayesian Network used in [3]
 					            P(X)= P(X6)P(X2/X6,X1)P(X3/X6).
            P(X4/X6,X3)P(X5/X2,X8)P(X1/X4,X7).
             P(X7/X4,X8). P(X8/X3)P(X9/X5,X7)

3	CEDAR-FOX Software

A system for forensic document examination was created as a result of US NIJ sponsored study on handwriting individuality known as CEDAR-FOX. It is a software created to assist writer identification and verification. This system consists of various methods such as writer verification, identification as well as signature matching. The features used for comparison include features(size, slant, spacing, shading, system, speed, strokes) commonly used by QD examiners alongwith micro and macro level features. CEDAR-FOX contains a number of user interfaces for QD examiner interaction and understanding of the system’s decision. Though comparison is largely done automatically, interfaces are provided for user to associate characters with correct decision before comparison is made.

We use the software in this project with a motive of handwritten image comparison for knowing if the image pair comes from same or different writer. The model used is the writer verification. The input provided is (1) the evidence: a scanned document whose writer is to be found,also known as questioned document, and (2) the know: a set of scanned documents that belong to the same writer. The output of this model is whether the questioned document and known documents belong to the same writer. This is represented in probabilistic terms as follows: Let x be feature representation for questioned document and y be the corresponding feature representation for the known document. If writer is same for questioned and known document then x=y and if not same then x≠ y. The log is taken of the probability of x and y when writer is same divided by probability of x and y when writer is different, also known as log likelihood ratio(LLR). If LLR is positive, then the questioned document and the known document have the same writer and if negative, then questioned document belongs to a different writer.   	

4	Convolutional Siamese Twin Network Model

Recent studies have shown a great deal of  performance rise in image recognition and classification using convolutional neural networks. CNNs are a form of feedforward neural network. In this project, we take up CNN as the deep learning approach for their ease of training.
We have used python language to implement the above described models. We have used a package, keras which makes implementation of all the layers involved in our twin architecture fairly simple and easy. Keras is a high-level neural network API, written in Python with tensorflow base for this project.

The base model for our project is a CNN architecture which has the following layers : a series of convolutional and max pooling layers which act as the feature extractors and a fully connected layer which performs non-linear transformations of the extracted features and acts as the classifier. For training, we used 1000 images and classified them according to 10 classes of writer. This image classification model gave us an accuracy of 86%. Now, instead of classifying, given two images we need to compute if the images are from same writer or different writer. 

So, we propose a siamese twin architecture with reference to a similar network proposed in [5] in which a twin model is used consisting of both CNNs i.e. our base model. The twins have the following model summary.

Table 3: Twin CNN model summary
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 126, 126, 8)       224       
_________________________________________________________________
batch_normalization_1 (Batch (None, 126, 126, 8)       32        
_________________________________________________________________
activation_1 (Activation)    (None, 126, 126, 8)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 124, 124, 16)      1168      
_________________________________________________________________
batch_normalization_2 (Batch (None, 124, 124, 16)      64        
_________________________________________________________________
activation_2 (Activation)    (None, 124, 124, 16)      0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 62, 62, 16)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 60, 60, 32)        4640      
_________________________________________________________________
batch_normalization_3 (Batch (None, 60, 60, 32)        128       
_________________________________________________________________
activation_3 (Activation)    (None, 60, 60, 32)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 58, 58, 32)        9248      
_________________________________________________________________
batch_normalization_4 (Batch (None, 58, 58, 32)        128       
_________________________________________________________________
activation_4 (Activation)    (None, 58, 58, 32)        0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 29, 29, 32)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 27, 27, 64)        18496     
_________________________________________________________________
batch_normalization_5 (Batch (None, 27, 27, 64)        256       
_________________________________________________________________
activation_5 (Activation)    (None, 27, 27, 64)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 25, 25, 64)        36928     
_________________________________________________________________
batch_normalization_6 (Batch (None, 25, 25, 64)        256       
_________________________________________________________________
activation_6 (Activation)    (None, 25, 25, 64)        0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 12, 12, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 9216)              0         
_________________________________________________________________
dense1 (Dense)               (None, 512)               4719104   
_________________________________________________________________
batch_normalization_7 (Batch (None, 512)               2048      
_________________________________________________________________
activation_7 (Activation)    (None, 512)               0         
_________________________________________________________________
output (Dense)               (None, 1024)              525312    
_________________________________________________________________
activation_8 (Activation)    (None, 1024)              0  =================================================================
Total params: 5,318,032
Trainable params: 5,316,576
Non-trainable params: 1,456

The image pairs are encoded using the above model. We get two vectors which are merged using chi square distance formula. :

	X2= (img1-img2)**2/img1+img2

Then, we pass through a linear classifier for obtaining the outputs. The siamese network, thus formed has the following model summary.

Table 4: Siamese Network Model Summary
Layer (type)                    Output Shape         Param #     Connected to                     
====================================================================input_1 (InputLayer)            (None, 128, 128, 3)  0                                            
_____________________________________________________________________________

input_2 (InputLayer)            (None, 128, 128, 3)  0                                            
_____________________________________________________________________________

sequential_1 (Sequential)       (None, 1024)         5318032     input_1[0][0]                    
                                                                 input_2[0][0]                    
_____________________________________________________________________________
merge_1 (Merge)                 (None, 1024)         0           sequential_1[1][0]               
                                                                 sequential_1[2][0]               
_____________________________________________________________________________
prediction (Dense)              (None, 2)            2050        merge_1[0][0] ====================================================================
Total params: 5,320,082
Trainable params: 5,318,626
Non-trainable params: 1,456

We get the output in form of two classes. The first class denotes the probability of the image pair coming from the same writer. The second class denotes a probability of image pair coming from a different writer. We use sigmoid activation function so that we get the output in the range of [0,1] as a probability. We use the binary cross entropy as our loss function as we need a binary classification with probability between [0,1]. It helps us measure the performance of  the model by comparing the predicted probability and actual labels. 

For training purpose, we create pair of images from same as well as different writer. As the number of different writer image pairs are far more than the number of same writer image pairs, a load balancing technique needed to be implemented such that there is a fair amount of dataset being trained for both the classes. We trained 83,153 image pairs consisting of 41,576 same writer pairs and 41,576 different writer pairs. With use of one epoch, we get accuracy as 78%. In the second epoch, we achieve 94% accuracy and the third epoch gives 97% accuracy.

Epoch 1/3
83153/83153[==============================]-9791s 118ms/step
- loss: 0.4551 - acc: 0.7830
Epoch 2/3
83153/83153[==============================]-9142s 110ms/step 
- loss: 0.1764 - acc: 0.9478
Epoch 3/3
83153/83153[==============================]-9690s 117ms/step
- loss: 0.0961 - acc: 0.9735

We tested 15,673 image pairs consisting of 7,836 same writer pairs and 7,836 different writer pairs. We achieved a fair amount of 77.5% accuracy on test data :
 
15673/15673 [==============================] - 661s 42ms/step
Test loss: 0.609404434527
Test accuracy: 0.775250430696

The model, as seen above, consists of 5,320,082 parameters, mostly belonging to the fully connected layer. We have a huge dataset due to usage of image pairs rather than the single images that is used for training. It helps us immensely in avoiding overfitting.

5	Summary and Discussion

We explored the three different techniques : Knowledge based approach, Simple Machine learning approach, Deep Learning approach for training the image pairs available. The use of python language in the project helps us take advantages of already existing packages such as pgmpy, keras, tensorflow etc. The project comprises of implementing already existing techniques and comparing the results to identify the technique that is more efficient in handwriting analysis.

In the project, we could achieve a 80.19% accuracy in the first model i.e by using a bayesian network proposed in [2]. With use of other bayesian networks architectures and more training samples, there is a scope for increasing the accuracy of this model. The second approach that consisted of use of a software based on simple machine learning approach CEDAR-FOX as described in [3] led to a accuracy of 65.7%. This accuracy was observed taking into account  70 image pairs. Other Simple machine learning techniques might yield a better accuracy as well. The third approach comprising of Siamese twin network resulted in an accuracy of 77.5%. With increase in depth of the CNN, we could achieve better results as we could observe of reduction of accuracy if we removed one of the convolutional layers in the twin network. 

We could see that with these basic models, Probabilistic graphical models perform more efficiently than a siamese twin network and the CEDAR-FOX software. With usage of more efficient models in all the three approaches, one can provide a clearer comparison among all these techniques. 

Acknowledgements

We thank Professor S. N. Srihari, Mihir Chauhan and Mohammad Abuzar Shaikh for all the useful guidance and  the data and software provided for the project.  



References
 [1] S. N. Srihari, S.-H. Cha, H. Arora, S. Lee, "Individuality of handwriting", J. Forensic Sci., vol. 47, no. 4, pp. 1-17, Jul. 2002.
[2] Mukta Puri, Sargur N. Srihari and Yi Tang: Bayesian Network Structure Learning and Inference Methods for Handwriting, Proc. Int. Conf. Document Analysis and Recognition, Washington DC, 2013.
[3] Srihari S.N.: Computational Methods for Handwritten Questioned Document Examination, US DoJ Report, December 2010.
[4] Sargur N. Srihari, Mukta Puri and Kirsten Singer: Samples of Cursively Written and Together with Characteristics and Probability
[5] Koch, G., Zemel, R., Salakhutdinov, R.: Siamese neural networks for one-shot image recognition. In: ICML 2015 Deep Learning Workshop (2015)
[6] Krizhevsky, A., Sutskever, I., Hinton, G.E.: Imagenet classification with deep convolutional neural networks. In: Neural Information Processing Systems (2012)
[7] François Chollet: https://keras.io
[8] Ankur Ankan, Abinash Panda: “pgmpy: Probabilistic Graphical Models using Python”. In Proceedings of the 14th Python in Science Conference (SCIPY 2015)
[9] F. Nielsen, R. Nock: " On the chi square and higher-order chi distances for approximating f -divergences ", IEEE Signal Process. Lett., vol. 21, no. 1, pp. 10-13, Jan. 2014.
[10] S.N. Srihari, H.L. Teulings: "A Survey of Computer Methods in Forensic Document Examination", Proc. of the 11th Conference of the International Graphonomics Society (IGS 2003), 2-5 November 2003, Scottsdale, Arizona, USA, pp. 278-281.
[11] S. N. Srihari, "Statistical Examination of Handwriting Characteristics using Automated Tools", US DoJ Report, pp. 85, April 2013.
