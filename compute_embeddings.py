from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
import numpy as np
from backbone.networks.inception_resnet_v1 import InceptionResnetV1
import torch

def get_embedding(model, face_pixels):

    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)

    samples = np.transpose(samples, (0,3,1,2))
    samples = torch.from_numpy(samples).float()

    yhat = model(samples)

    return yhat[0]

data = load('5-student-faces-dataset.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

model = InceptionResnetV1(pretrained='vggface2')
model.load_state_dict(torch.load("experiments/group22.pt", map_location=torch.device('cpu')))

print('Loaded Model')

newTrainX = list()
for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding.detach().cpu().numpy())
newTrainX = asarray(newTrainX)
print(newTrainX.shape)

newTestX = list()
for face_pixels in testX:
	embedding = get_embedding(model, face_pixels)
	newTestX.append(embedding.detach().cpu().numpy())
newTestX = asarray(newTestX)
print(newTestX.shape)

savez_compressed('5-student-faces-embeddings.npz', newTrainX, trainy, newTestX, testy)