#MAY 2018
#CREATING A CUSTOM CORE ML MODEL USING PYTHON AND TURI CREATE
#Install turi create using  pip install turicreate

import turicreate as turi
url = "dataset/"

#find and load images from the dataset folder
data = turi.image_analysis.load_images(url)

# We define image categories based on its folder path:
data["foodType"] = data["path"].apply(lambda path: "Rice" if "rice" in path else "Soup")
#save the new labeled dateset as rice_or_soup.sframe
# We will use it to train the model
data.save("rice_or_soup.sframe")

#Preview the new labeled dataset on Turi Create
data.explore()

##########   Training & Exporting the Machine Learning model  ###################
# note: There are 2 different architectures to train- SqueezeNet & ResNet-50 (more accurate)

# load the previously saved rice_or_soup.sframe file
dataBuffer = turi.SFrame("rice_or_soup.sframe")

# create training data using 90% of the dataBuffer object we just created and test data using the remaining 10%.
trainingBuffers , testingBuffers = dataBuffer.random_split(0.9)

# create the image classifier using the training data and SqueezeNet architecture:
model = turi.image_classifier.create(trainingBuffers, target="foodType", model="squeezenet_v1.1")

# evaluate the test data to determine the model accuracy:
evaluations = model.evaluate(testingBuffers)
print (evaluations["accuracy"])

#save the model and export the image classification model as a CoreML model:
model.save("rice_or_soup.model")
model.export_coreml("RiceSoupClassifier.mlmodel")


