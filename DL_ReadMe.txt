Instructions for testing hidden test data :
	
	1. Please give the location of images in variable AND_hidden_Dataset_path in line num 211
	2. Please give the csv filename of pairs in line 232
	3. For loading wights of already trained model : Line 155,156
	4. Please run codes line 205-260 (extracting images and predicting and printing output in csv file)
	5. Please find the output in file CNNTestOutput.csv containing columns : FirstImage, SecondImage, Same or Different				

The code consists of the following:

1. Importing the necessary library and packages

2. Read images from the specific folder path using OpenCV
		t_x : Image array consisting of all images
		t_y : Labels array consisting of the specific writer number to whom the image belongs

3. Splitting the dataset into train and test data in 4:1 ratio
		train_X : Image array for training
		train_Y : Training Label array 
		test_X  : Image array for testing 
		test_Y  : Testing Label array 

4. Creating the CNN model with following configuration :
		Input shape fixed 	: (128,128,3)
		Model 			: Sequential
		Layers 			: 1.	Conv2D, BatchNormalization, ReLu Activation
					  2.	Conv2D, BatchNormalization, ReLu Activation
					  3.	MaxPooling2D
					  4.	Conv2D, BatchNormalization, ReLu Activation
					  5.	Conv2D, BatchNormalization, ReLu Activation
					  6.	MaxPooling2D
					  7.	Conv2D, BatchNormalization, ReLu Activation
					  8.	Conv2D, BatchNormalization, ReLu Activation
					  9.	MaxPooling2D
					  10.	Flatten
					  11.	Dense, BatchNormalization, ReLu Activation
					  12.	Dense, Sigmoid Activation
		Encoded both images as encoded_left, encoded_right
		Merge two images using : Mode: Chi Square Distance
		Dense layer : Linear classification into Same or Different class
		Optimizer used : Adam
		Learning Rate : 0.00006
		Loss function used : Binary Cross entropy

5. Creating image pairs for training and balancing the load for same and different writer(~80k pairs)
		train_img_1 : Image array containing first image   
		train_img_2 : Image array containing second image
		train_label : 1st column : Same		   (Values : 0 if not same 1 if same)
			      2nd column : Different 	   (Values : 0 if not different 1 if different)
		train_y : one writer name if same writer
			  two writer names with a space between 2 numbers if different writer

6. Training the data by passing list containing train_img_1 and train_img_2 as inputs
		y = train_label
		batch size : 50 
		epoch = 3
 		Results : 
			Epoch 1/3
			83153/83153 [==============================] - 9791s 118ms/step - loss: 0.4551 - acc: 0.7830
			Epoch 2/3
			83153/83153 [==============================] - 9142s 110ms/step - loss: 0.1764 - acc: 0.9478
			Epoch 3/3
			83153/83153 [==============================] - 9690s 117ms/step - loss: 0.0961 - acc: 0.9735

7. Train accuracy and loss observed :  

8. Saved weights : File name : "weights-project-CNN-2.hdf5" Line 155,156 show the code to load saved weights for the CNN model created above. 

9. Creating image pairs for testing and balancing the load for same and different writer(~20k pairs)
		test_img_1 : Image array containing first image
		test_img_2 : Image array containing second image
		test_label : 1st column : Same		   (Values : 0 if not same 1 if same)
			      2nd column : Different 	   (Values : 0 if not different 1 if different)
		test_y : one writer name if same writer
			 two writer names with a space between 2 numbers if different writer

10. Test Accuracy and loss observed : 



11. For testing hidden test data :
	1. Please give the location of images in variable AND_hidden_Dataset_path in line num 211
	2. Please give the csv filename of pairs in line 232
	3. Please find the output in file CNNTestOutput.csv containing columns : FirstImage, SecondImage, Same or Different
				


		
		
