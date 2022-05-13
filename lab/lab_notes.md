# Deep learning lab

## 1 b)

I choose to start with a size of 32x32 because of three reasons

### Reason 1

[When I did a small eda](https://github.com/felix-tjernberg/ai21-deep-learning/blob/main/lab/1_image_processing.ipynb) I found out that the smallest side in the dataset was 32px

I decided at that point to scale down everything to 32x32px so all the files have the same amount of data.

[After doing a lot of hyper parameter tuning](https://github.com/felix-tjernberg/ai21-deep-learning/blob/main/lab/2_32x32_models.ipynb) I realized that 32x32px was to little information to get a really good prediction, I maxed out at around .73 val_acc which isn't very good.

One thing that was notable for this image size was that a smaller kernel size of 2x2 gave a somewhat better result at .74, which might be suggest that giving the model got some more information with a smaller stride and kernel size

### Reason 2

I also wanted to see how little information you could give to a model to get a decent inference

[Also tried 64x64px](https://github.com/felix-tjernberg/ai21-deep-learning/blob/main/lab/2_64x64_models.ipynb) but it did not really make much of a difference, something I also tried during this time was to see how much difference the transformations helped. It seemed to do quite a lot for this image size, this theory was also reinforced when talking to a classmate as they dit not have so much of a jump in performance from transformations

### Reason 3

I also wanted really fast training times when I did hyper parameter tuning for this project, I also had this hypothesis: as it's only two classes I suspect the algorithm will figure out a key difference between the classes and then just search for that difference in the picture;my guess would be that the feature will be the triangular ears of the cats

This was later switch to 64x64 later on

## 1 d)

I choose the parameters for augmentations quite arbitrarily

Something I did realize when talking to classmates is that havning a rotation range of up to 90 degrees might be quite high as they seemed to get worse models when having that high of degree rotation.

So I'm quite curious what my models learned as they seemed to like the high rotation range

## 2 a)

[When I tried to do the hyperparameter tune my first models](https://github.com/felix-tjernberg/ai21-deep-learning/blob/main/lab/2_32x32_models.ipynb) I tried to follow the rules of thumb from [this article](https://towardsdatascience.com/17-rules-of-thumb-for-building-a-neural-network-93356f9930af), in my case hyper parameter tuning did not help much: which might be because the model did not get enough information from the beginning **_"Shit in Shit out"_**

I did talk to my teacher and he said that hyperparameter tuning doesn't usually give you more than a few more percent in extra performance so long as you choose a decent network architecture to start with

So my take away from this lab is that hyperparameter tuning of a deep learning model should only be done a small amount.

If you still have a bad result you should probably start rethinking if your input data is good enough for your expected result or if you have really chosen the correct type of model for the task at hand: this thought was also reinforced when I tried transfer learning

If I did this lab again I would do more hyperparameter tuning on the image augmentation instead of the mlp model to avoid **_"Shit in Shit out"_**

## 2 c)

### Hyper parameter choises for mlp model

I choose the 64x64px model as it seemed to do a little bit better than all the 32x32px models

Choose training 137 epochs as it's a cool number and around that point the [64x64px model got score 75% in validation accuracy](https://github.com/felix-tjernberg/ai21-deep-learning/blob/main/lab/2_64x64_models.ipynb)

[Getting 70% accuracy](https://github.com/felix-tjernberg/ai21-deep-learning/blob/main/lab/2_model_selection_and_inference.ipynb) on the test data is actually quite decent for a model that only takes images that is 64x64px large :)

## 2 d)

### Hyper parameter choices for transfer learning

For transfer model I choose Xception as feature extractor because it seemed to be a big strong boi :)

#### Fast feature extraction method vs Combined feature extractor + model method

I choose **fast feature extraction method** because it makes training the full model much quicker

1.  Doing fast feature extraction means that you do a prediction on your training dataset with a pre trained model and then train rest of the model on those features maps (predictions)

    This is done because in most cases you do not want to retrain the feature extractor part (pre trained model) of the model and you freeze these weights

    Which means that you only need to run the pre trained part once on the training dataset instead of each step when training the whole model

    One thing to note here is that when doing the predictions from the pre trained model you have to add one last layer that flattens the kernels into a more digestible 2 dimensional matrix/tensor, om the combined method this is also done to glue the both parts together

    In this project I tried both **GlobalAveragePooling2D** and **Flatten** layers to create the 2 dimensional outputs

    I did not see much of a difference between GlobalAveragePooling2D and Flatten in prediction strength so I choose GlobalAveragePooling2D because that generated a smaller matrix `(sample_number, 2048) vs (sample_number, 204800)` which made training orders of magnitude faster

2.  Doing fast feature extraction also means that you can use normal machine learning models as predictors and not try to approximate a classification function with a mlp model

3.  You do not have to load the feature extraction part of the model in memory when doing training which speeds up the training time by orders of magnitude

#### Other hyper parameters and result

For weights I used imagenet. ( After evaluation I started to wonder if imagenet includes the images from the dataset we were provided with in this lab: in effect data leakage )

[When I tried a mlp model the results got quite good with (99.4 % validation accuracy)](https://github.com/felix-tjernberg/ai21-deep-learning/blob/main/lab/2_transference_feature_map_method.ipynb)

But a classmate is supersold on RandomForestClassifier and asked me to try it with the fast feature extraction method and I got a crazy good 99% validation accuracy!

This then became my choice for training on training and validation data: Xception as the feature extractor and RandomForestClassifier as classifier

[The results speaks for itself](https://github.com/felix-tjernberg/ai21-deep-learning/blob/main/lab/2_model_selection_and_inference.ipynb), only 8 misclassified images!

### Xception summary
