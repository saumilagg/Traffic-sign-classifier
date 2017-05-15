#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/visualization1.png "Visualization"
[image2]: ./examples/histo.png "Histogram"
[image3]: ./traffic_signs/s1.jpg "Traffic Sign 1"
[image4]: ./traffic_signs/s2.jpg "Traffic Sign 2"
[image5]: ./traffic_signs/s3.jpg "Traffic Sign 3"
[image6]: ./traffic_signs/s4.jpg "Traffic Sign 4"
[image7]: ./traffic_signs/s5.jpg "Traffic Sign 5"
[image8]: ./traffic_signs/s6.jpg "Traffic Sign 6"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

You're reading it! and here is a link to my [project code] (https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Basic summary of the data set.

I used numpy library to calculate the summary of dataset. The follwing snipets show how this was done:
* The size of training set is len(X_train)
* The size of the validation set is len(X_valid)
* The size of test set is len(X_test)
* The shape of a traffic sign image is X_train[0].shape
* The number of unique classes/labels in the data set is np.unique(np.concatenate((y_train,y_valid,y_test))).size

Note: X_train, X_valid & X_test are the train, valid and test data respectively.

####2. Include an exploratory visualization of the dataset.

For exploratory visualization of dataset, I used matplotlib to draw bar graph for my training data set. To get a further clear understanding I drew a histogram for my test data to see the exact distribution and what can be done in terms of augmentation methods.

*Bar graph for training dataset:

![alt text][image1]

As we can see the training data is quite skewed with very less number of samples for some label's. It is possible that the predicition would not be as accurate as desired.

*Makes me wonder how test data is distributed. To look more closely, I plotted a histogram for test data

![alt text][image2]

###Design and Test a Model Architecture

####1. Preprocessing the image data.

*First, I converted image to grayscale as we are using Le-Net architecture and the example from MNIST dataset where the images were 28x28 grascale; I choose to conver the images first. Also, it would be easier to predict the correct sign with varying color intensities after converting an image to grayscale. I have printed out the shape so as to cofirm and test the images are converted. Ref : In [9] shape (32,32)

*Also, while converting to grayscaleI normalized the image data so that the data has zero mean and equal variance. My normalization code looks like this :

image = image/255. - 0.5  #normalize the image data

*Then, I flattened the image to convert into a 2D array

####2. Final model architecture

 Used the LeNet architecture.
 
 My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, 'VALID' padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 3x3	    | 1x1 stride, 'VALID' padding, outputs, 10x10x16 |
| RELU                  |												|
| Max pooling           | 2x2 stride, outputs 5x5x16, 
| Flatten 				| Input 5x5x16 , outputs 400 
| Fully connected		| input 400, outputs 120      					|
| RELU                  | 												|
| Dropout				| Since the layer is fully connected and has           all 					   features/parameters; it is beneficial to apply dropout at this point	|
| Fully Connected		| input 120, output 84			                |
| RELU                  | 												|
| Fully Connected       | Input 84, outputs 43						    |
 
 
 ####3. Training the model.
 
 To train the model, I used a Batch size of 128, Epochs 50, a learning rate of 0.001. In addition to this I used an Adam Optimizer for first-order gradient-decent optimization which computes adaptive learning rates for each parameter. 
 
 ####4. Solution Approach.
 
 My final model results were:

 * validation set accuracy of  0.964
 * test set accuracy of 0.936
 
 Well, I used the well known Le-Net architecture
 
 *What architecture was chosen?
 
 Le-Net architecture was adopted for this model. This is a convolution neural network used for digit recognition. The current model was used for OCR and character recognition in documents. Adding to this it being starightforward and small makes it easy to use without much resources and hence a good option to start with for traffic sign recognition.
 
 *Why did you believe it would be relevant to the traffic sign application?
 
 According to the Yann paper on LeNet architecture, he mentions that it is specifically designed for handwritten and machine printed character recognition. And we have already tested it on MNIST dataset. The robustness of the model and the CNN designed to recognize visual paterns from image pixels with minimal preprocessing makes it a good starting point for traffic sign application.
 
 *How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
 After training the model on train data and checking across valid data we see that we have achieved an accuracy of 96.4%. That shows that our model should be able to correctly predict the sign with such accuracy. And that too with just image preprocessing and dropout methods. However when we tested on a test data the accuracy droped to 93.6%. This was because of the skewed distribution of dataset. Other techniques like augmenting the data, image rotation, etc might help achive better accuracy and I would work on it in the later versions of the project!
 
###Test a Model on New Images

####1.Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7] ![alt text][image8]

The 1st, 5th & 6th images would be difficult to recognize as the signs have some kind of text at the bottom. This might confuse the model as its not trained on such images types.
 
####2.Model's predictions on these new traffic signs
 
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Child Crossing      	| Bicycles Crossing   							| 
| 60 km/h     			| 60 km/h   									|
| Right of way at intersection	| Right of way at intersection			|
| Road Work	      		| Road Work					 				    |
| Caution			    | No passing      							    |
| Bumpy Road            | No passing									|

The model was able to correctly guess 3 of the 6 traffic signs, which gives an accuracy of 50%. However, the accuracy on the test set was much higher which indicates that the model could be trained much better.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.
 
The code for making predictions on my final model is located in the 22nd cell of the Ipython notebook.
 
First image

![alt text][image3]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Bicycle crossing 								| 
| .001     				| Road Narrows on right	    					|
| .00005				| Speed Limit 30km/h							|
| .00002	      		| Children Crossing				 				|
| .000002			    | General Caution      							|
 
 
Second image

![alt text][image4]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed 60km/h   								| 
| .00004     			| Speed 50km/h									|
| ~0				    | Speed 80km/h									|
| ~0	      			| Speed 30km/h					 				|
| ~0				    | Right of way at intersection					|

Third image

![alt text][image5]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0    			    | Right of way at intersection					| 
| ~0	   				| Pedestrains									|
| ~0 					| Double Curve	    							|
| ~0 	      			| Beware of ice snow			 				|
| ~0 				    | Road narrows on the right					|

Fourth image

![alt text][image6]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .80       			| Road Work										| 
| .19    				| Bumpy Road									|
| ~0					| Bicycle Crossing								|
| ~0  					| Keep Left						 				|
| ~0				    | Wild Animals Crossing	 						|

Fifth image

![alt text][image7]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No Passing   									| 
| ~0     				| Turn Left Ahead								|
| ~0					| Roundabout mandatory							|
| ~0	      			| Slippery Road					 				|
| ~0				    | End of No passing								| 

Sixth image

![alt text][image8]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| No passing 									| 
| 0.0001     			| End of passing								|
| ~0					| Slippery Road									|
| ~0	      			| Vehicles over 3.6 metric tons prohibited 		|
| ~0				    | Ahead only									| 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 