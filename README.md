# COMPARATIVE-STUDY-ON-FIREARM-DETECTION-USING-CONVOLUTIONAL-NEURAL-NETWORK


Dataset:https://drive.google.com/drive/folders/1DgKzqEjS26qq1YBlWDyzpKAH2Ca7avoh?usp=share_link

Abstract

This project is to train a model to detect different firearms using convolutional neural network (CNN). The model will be trained in different architectures of CNN. This will give us information on which model shows better performance in terms of accuracy and detection speed. 
The accuracy and speed of detection are crucial factors in the event of firearm detection as how fast and accurate the firearm can be identified makes the true difference in saving lives. Different CNN architectures can give different performance outputs not just for dissimilar problem statements, but for the same problem statements too. The complexity of the architecture can be a differentiating factor when choosing speed as the desirable outcome. But complex architectures might be able to produce more accurate results. So, choosing the right parameters to assess the output to obtain the optimal balance between speed and accuracy is essential when developing a deep learning solution in this context. The best architectural approach to solve this problem is critical for the desired result. 

Introduction 

Firearms are deeply integrated into America’s social and political wirings. The right to possess and carry a firearm is every person’s constitutional right. Being said that this device is extremely dangerous when it is in the hands of individuals who choose not to follow the rules. There are many gun-free zones in the USA where carrying a firearm is prohibited by law. The staggering statistics show that 1 in 4 households has firearms in the us. Even with this level of gun ownership almost half of Americans think that gun violence is a problem in this country. Detecting firearms can also be a helpful data point in the event of any public chaos. Quick and accurate detection of firearms to limit gun usage in prohibited zones or to document as a data point in general in any public event can be a positive reinforcement for the law-abiding gun owners too. 
When someone is holding a holstered firearm in public it is legitimate for the security staff to be aware and alerted. Some states do not allow open carry. Under state brandishing and assault laws carrying firearm in open site is typically illegal. It is not even a question that security staff need to be alerted when this happens at firearm-free zones like schools, federal buildings, airports, hospitals, bars, and taverns, etc. Many legal gun owners in this country are in full-fledged support of early identification of people who are not worthy of possessing a firearm. If we can identify people who don’t hesitate to do a public display of firearms, especially in gun free zones needs to be revalidated to make sure that the gun ownership is properly vetted and is adhering to laws of the land. This method of firearm detection can be less intrusive and can evoke less emotion driven response on security checks.  
With these very real and impactful use cases I am keen to apply my deep learning academic skills to conclude on an architecture that will provide the right solution. By doing this I can get hands on experience on different architecture which can be used in different problem statement in the future.

Problem 

This project is to train a model to detect different firearms using convolutional neural network (CNN). The model will be trained in different architectures of Convolutional Neural Networks. This will give us information on which model shows better performance in terms of accuracy and detection speed. 
The accuracy and speed of detection are crucial factors in the event of firearm detection as how fast and accurate the firearm can be identified makes the true difference in saving lives. Different CNN architectures can give different performance outputs not just for dissimilar problem statements, but for the same problem statements too. The complexity of the architecture can be a differentiating factor when choosing speed as the desirable outcome. But complex architectures might be able to produce more accurate results. So, choosing the right parameters to assess the output to obtain the optimal balance between speed and accuracy is essential when developing a deep learning solution in this context. The best architectural approach to solve this problem is critical for the desired result. 

Implementation

Preprocessing of the data
To assure the model's accuracy in this project, a variety of data cleaning and preprocessing procedures are employed. First, using the Python hashlib library, duplicate images in the datasets for knives and firearms are checked. The file paths and image hashes are saved in a dictionary. De-duplication of images is performed in the image library of knives and firearms. This is to make sure that the dataset will not contain the same image more than once.

Using the ‘sci-kit-learn’ train_test_split technique, the folders are then divided into training and testing sets. To prevent dataset bias, the images in both datasets are swapped at random. The images are then divided into training and testing sets with a ratio of 0.2 for testing and 0.8 for training. The photos for the training and testing for knives and firearms are then copied to the appropriate folders. The folders are labeled folders respectively for the model to classify on these sets. 
Various preprocessing methods are then used to get the data ready for training on different models. The photos are normalized by dividing each pixel value by 255 and scaled to 224 by 224 pixels. The labels for the images are encoded as one-hot vectors with each element representing a class and being either 0 or 1 depending on whether the image corresponds to that class or not. This is done with the class_mode set to "categorical." Cleaning and preprocessing of the data ensure that the dataset is unbiased and reliable so the model can be trained and produce the results accurately.

Training the models

The dataset was trained on three different architectures of CNN which are VGG16, ResNet-50, and MobileNet. All the necessary libraries required for accomplishing this purpose were imported. Image size, batch size, number of epochs, and initial learning rate were set to (224, 224), 32, 10, and 0.001 respectively and the data is then preprocessed. All the parameters are kept the same except the pre-trained model used for training is changed to see the change in results. 
The pre-trained models are loaded, and a few modifications have been made to them, additionally a GlobalAveragePooling2D layer and two Dense layers with the activation function 'relu' and 'SoftMax' respectively added.  " The optimizer that is used is SGD with an initial learning rate of 0.001 and a momentum of 0.9.  categorical_crossentropy is used as the loss function. The model is trained using the fit method by passing the training and testing data generators along with the required callbacks such as LearningRateScheduler, ModelCheckpoint, and EarlyStopping. "     

When pre-defined conditions are fulfilled, the ModelCheckpoint and EarlyStopping callbacks are used to take specific actions while the model is being trained. This is used to keep track of the model's performance. The weights of the model with the highest validation accuracy are saved throughout training by the ModelCheckpoint callback. This is advantageous because the model's weights can be used as an inference or to pick up where the training process left off. The EarlyStopping function terminates training early if the model's validation accuracy does not increase for a predetermined period of epochs and if the value does not change to a minimum amount. If the model's performance has plateaued and is not anticipated to get much better, doing this can save time and computational resources.
The usage of both callbacks can assist the model from overfitting as well as ensuring the model’s weights are saved at the point of greatest validation accuracy.   

Testing the models.

The testing dataset, which comprises of two classes, namely knives and firearms, was used to test each of the trained models. Two essential criteria of accuracy (“which is the fraction of correctly classified images”) and execution time (“which is the time required to classify the images”) to evaluate the dataset's performance are used. Also, an additional image was fed that was absent from the dataset and input two random images from the testing dataset, for the model to classify them. The results were compared to determine which model performed the best in terms of accuracy and execution time.
3.2.3	Training and testing the models on augmented dataset.
Augmentation technique is applied to increase the data set to measure any notable improvement to the performance of these models. The augmentation is applied on both the training and testing dataset. After that MobileNet and ResNet were trained again but this time on the augmented dataset. Then these models are tested using the augmented testing dataset, and the outcomes were contrasted to establish which model executed more quickly and accurately.

Methodology, Results and Analysis 

Methodology 

The accuracy and speed of detection are crucial factors in the event of firearm detection as how fast and accurate firearms can be identified makes the true difference in saving lives. The 2 classes of testing datasets that are used in this project are Knives and firearms. Accuracy in this context is defined as ‘the fraction of correctly classified images. The execution time is defined as “the time required to classify the images.”
The model is fed with two images from the randomly selected testing dataset. An additional random image is also given as input. The results were compared with the above defined output parameters. 
 


Results

Number of Images in training dataset: 6094
Number of Images in testing dataset: 1524

VGG16

Training
According to information on the model training utilizing the VGG16 architecture, the training was completed in five training epochs. The model was trained over 191 steps (batches)per epoch, with an average training time of 22 seconds per step. Throughout training and validation, the metrics for training loss and accuracy were kept track of. The first epoch's training accuracy was 54%, while the fifth epoch's accuracy was 92.52%. From 1.51 in the first epoch to 0.20 in the fifth, the training loss decreased. The validation accuracy changed between 61.55% and 95.01% in the following three epochs after beginning at 53.41% in the first epoch and increasing to 95.01% in the second. From 1.02 in the first epoch to 0.14 in the fifth, the validation loss decreased. A learning rate of 0.001 was used to train the model, and it remained constant during training. The validation accuracy in the fifth epoch was 0.9587. The model was saved after each epoch using the best validation accuracy attained thus far. Because the validation accuracy did not increase in the fourth epoch, the training was terminated early after the fifth one. With high training accuracy and a minimal training loss, this model appears to perform well. However, the validation accuracy fluctuated, which may indicate overfitting. 

  

Testing
After testing, the test accuracy was 0.9586614370346069 and test loss was 0.13998650014400482. Also, the time taken for predicting the image was 0.626527786254 seconds.


MobileNet

Training: The training accuracy and validation accuracy for the first epoch were 96.16% and 99.80%, respectively. In the first epoch, the validation accuracy increased from a negative infinity to a value of 99.80%. The validation accuracy improved to 99.87%, and the training accuracy went to 98.85% in the second epoch. In the third epoch, the training accuracy was 99.03%, and the validation accuracy was 99.87%. The model was not saved since the validation accuracy did not improve. The training accuracy climbed to 99.13% in the fourth period, while the validation accuracy reached 99.87%. As the validation accuracy did not increase, the training was terminated early in the fourth epoch. The saved model's test accuracy was 99.87% and its test loss was 0.0055. 

  
Training Epochs vs Training Loss and Accuracy of MobileNet Model

Testing: After testing, the test accuracy was 0.9986876845359802 and test loss was 0.005464654415845871. Also, the time taken for predicting the image was 0.08080244 seconds. 

 

 ResNet
Training: In the first epoch of training, the model had an accuracy of 97.78% and a loss of 0.0602. The model improved its accuracy to 99.49% and 99.85% in the succeeding epochs, respectively, with a smaller loss of 0.0170 and 0.0051. The model's validation accuracy did not increase in the fourth epoch, however, and it halted at 99.95% accuracy with a loss of 0.0027. The model was trained with a learning rate of 0.0010 and early stopping was applied. The early stopping mechanism terminated the training process in the fifth epoch because the validation accuracy did not increase after the fourth epoch. The model dramatically increased in validation accuracy during the epochs, going from 84.65% to 99.93%. In the later epochs, the model had a higher validation accuracy, demonstrating that it was improving and learning as it went along. The fact that there is not much of a difference between the accuracy during training and validation shows that the model did not overfit the training data and that it was also able to generalize well. 



Testing: After testing, the test accuracy was 0.9993438124656677 and test loss was 0.004195770714432001. Also, the time taken for predicting the image was 0.152593851 seconds.

 




With Augmentation
Number of Images in training dataset: 24376
Number of Images in testing dataset: 6096

MobileNet with Augmentation  

Training: After four epochs, the training MobileNet was completed. The model attained a training accuracy of 0.9630 and a training loss of 0.0978 in the first epoch. The validation loss was 0.0140 and the validation accuracy was 0.99623. The training accuracy increased to 0.9877 and the training loss fell to 0.0360 in the second epoch. The validation loss was 0.0138 and the validation accuracy increased to 0.99672. As the validation accuracy increased, the best model was saved. The training accuracy climbed to 0.9937 and the training loss dropped to 0.0197 in the third period.
The validation loss fell to 0.0085, while the validation accuracy increased to 0.99803. As the validation accuracy increased, the best model was once more saved. The training accuracy increased marginally to 0.9962 and the training loss dropped to 0.0116 in the fourth epoch. However, the model was not retained as the best model because the validation accuracy did not increase from the previous iteration. Given that the model has attained a high degree of accuracy, the training procedure was terminated early. The model was able to obtain a high degree of accuracy overall on the validation data, showing that it can generalize to new data with ease. The model was successfully learning the task as the training loss dropped continuously over the epochs.


Testing: After testing, the test accuracy was 0. 9977034330368042 and test loss was 0.006670986767858267. Also, the time taken for predicting the image was 0.15696501 seconds.

ResNet with Augmentation  

Training: The model had a training accuracy of 0.9744 and a validation accuracy of 0.9806 in the initial epoch. The validation loss was 0.056, while the training loss was 0.0689. The model was saved to the file with the highest validation accuracy. The model performed better in the subsequent epoch, reaching training accuracy of 0.9954 and validation accuracy of 0.9962. The validation loss was 0.0105, while the training loss was 0.0127. The model was once more saved to the same file with the highest validation accuracy. The model kept getting better in the third epoch, reaching training accuracy of 0.9976 and validation accuracy of 0.9969. The validation loss was 0.0062, while the training loss was 0.0066. With the best validation accuracy once more, the model was stored. The model attained a training accuracy of 0.9986 and a validation accuracy of 0.9975 in the fourth epoch. The validation loss was 0.0073, while the training loss was 0.0049. The model with the highest validation accuracy was saved. The model considerably enhanced its performance in the fifth epoch, reaching training accuracy of 0.9993 and validation accuracy of 0.9977. The validation loss was 0.0067, while the training loss was 0.0026. With the best validation accuracy once more, the model was stored. The training was stopped early in the fifth epoch since the validation accuracy did not improve further.

  

Testing: After testing, the test accuracy was 0. 006670986767858267 and test loss was 0.006670986767858267. Also, the time taken for predicting the image was 0.08383750 seconds. 

 
  Analysis 
 
 
![image](https://user-images.githubusercontent.com/65331027/236603309-4f824689-810c-4f64-8112-c062bbd727df.png)

![image](https://user-images.githubusercontent.com/65331027/236603325-023bc5d9-fbc3-4e95-a92d-d446e0d64755.png)

![image](https://user-images.githubusercontent.com/65331027/236603333-9f9c3bcd-fd29-4bc7-b7f9-15aa2bb64a90.png)


![image](https://user-images.githubusercontent.com/65331027/236603339-fd017ac5-b84c-41bd-8713-7717c5911630.png)


All models performed exceptionally well in firearm detection measured on accuracy, with ResNet and MobileNet showing slightly better accuracy than VGG16. MobileNet with augmented dataset and ResNet with augmented dataset also performed very well, although their accuracy was slightly lower than that of the original MobileNet and ResNet models.
When test loss is measured, ResNet and MobileNet with augmented dataset performed slightly better than the other models. This is an indication that they were able to generalize better and were less prone to overfitting the training data.
In terms of execution time, MobileNet was the fastest model, with the original MobileNet and MobileNet with augmented dataset showing very similar execution times. VGG16 and ResNet models took more time for prediction than MobileNet, although the difference in execution time was not substantial. ResNet with augmented dataset took slightly more time for prediction than the original ResNet model.

Conclusions

All the models showed impressive performance in firearm detection, but ResNet and MobileNet performed slightly better than VGG16. MobileNet was the fastest model, while ResNet and MobileNet with ‘augmented dataset’ showed slightly better generalization ability than the other models. The choice of the best model will depend on the specific use case and the tradeoff between accuracy and execution time. It is true that VGG16 is generally slower than ResNet and MobileNet in terms of execution time per epoch. This is because VGG16 has a much deeper architecture, which requires more computations and memory usage. In comparison, ResNet and MobileNet use skip connections and depth wise convolutions respectively, which can reduce the number of parameters and computations needed for training and inference. However, the execution time also depends on the specific hardware and software configuration used for training, as well as the size and complexity of the dataset.

Summary

As part of this academic Project, the performance of three popular convolutional neural network models is comapred, VGG16, ResNet, and MobileNet, for firearm detection. A dataset of images containing firearms and knives are used, and compared the models based on their test accuracy, test loss, and execution time. Based on the analysis, we found that all the models performed very well in firearm detection, with ResNet and MobileNet showing slightly better accuracy than VGG16. The original MobileNet and ResNet models achieved the highest test accuracy, while VGG16 had the lowest accuracy. When trained with augmented data, MobileNet and ResNet also showed impressive performance, although their accuracy was slightly lower than that of the original models. When test loss was assessed, it is found that ResNet and MobileNet with augmented dataset performed slightly better than the other models, that suggests they were less prone to overfitting the training data and had better generalization ability. In terms of execution time, MobileNet was the fastest model, while VGG16 and ResNet took more time for prediction than MobileNet. Overall, our comparative analysis suggests that ResNet and MobileNet are good choices for firearm detection tasks, as they achieved high accuracy and low-test loss. The choice of the best model will depend on the specific use case and the tradeoff between accuracy and execution time.

