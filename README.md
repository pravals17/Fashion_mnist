# Fashion_mnist
Classifying Fashion mnist using a simple ANN
This deals with the classiﬁcation of Fashion MNIST dataset using artiﬁcial neural network in Python TensorFlow. The dataset contains 60,000 images for training and validation and 10,000 images for testing. Each image is a 28x28 grayscale image which is ﬂattened into an array of 784 pixels. All the images belong to one of the 10 classes as listed below:
• 0 T-shirt/top
• 1 Trouser 
• 2 Pullover 
• 3 Dress 
• 4 Coat 
• 5 Sandal 
• 6 Shirt 
• 7 Sneaker 
• 8 Bag 
• 9 Ankle boot

The files model.py creates a tensorflow model, util.py contains additional fucntions that are useful for data preprocessing such as one-ho-encoding, main.py implements the fashion mnist classification using a train, test and validation fold whereas main1.py implements a k-fold cross validation for the classification. An accuracy of 88% is obtained implementing both methods. 
