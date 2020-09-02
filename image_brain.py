# prior to running this, please install required dependencies for ImageAI, refer to https://github.com/OlafenwaMoses/ImageAI

from imageai.Prediction import ImagePrediction
import os
execution_path=os.getcwd()

prediction = ImagePrediction()
prediction.setModelTypeAsSqueezeNet()
# select and download the model of your choice and have it saved in the same location as this file
# here I am using the SqueezeNet model
prediction.setModelPath(os.path.join(execution_path, "squeezenet_weights_tf_dim_ordering_tf_kernels.h5"))
prediction.loadModel()

# use image of your choice here
predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "your image of choice.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)