Triplet loss for facial recognition.

# Triplet Face

The repository contains code for the application of triplet loss training to the
task of facial recognition. This code has been produced for a lecture and is not
going to be maintained in any sort.

![TSNE_Latent](TSNE_Latent.png)

## Architecture

The proposed architecture is pretty simple and does not implement state of the
art performances. The chosen architecture is a fine tuning example of the
resnet18 CNN model. The model includes the freezed CNN part of resnet, and its
FC part has been replaced to be trained to output latent variables for the
facial image input.

The dataset needs to be formatted in the following form:
```
dataset/
| test/
| | 0/
| | | 00563.png
| | | 01567.png
| | | ...
| | 1/
| | | 00011.png
| | | 00153.png
| | | ...
| | ...
| train/
| | 0/
| | | 00001.png
| | | 00002.png
| | | ...
| | 1/
| | | 00001.png
| | | 00002.png
| | | ...
| | ...
| labels.csv        # id;label
```

## Install

Install all dependencies ( pip command may need sudo ):
```bash
cd TripletFace/
pip3 install -r requirements.txt
```

## Usage

For training:
```bash
usage: train.py [-h] -s DATASET_PATH -m MODEL_PATH [-i INPUT_SIZE]
                [-z LATENT_SIZE] [-b BATCH_SIZE] [-e EPOCHS]
                [-l LEARNING_RATE] [-w N_WORKERS] [-r N_SAMPLES]

optional arguments:
  -h, --help            show this help message and exit
  -s DATASET_PATH, --dataset_path DATASET_PATH
  -m MODEL_PATH, --model_path MODEL_PATH
  -i INPUT_SIZE, --input_size INPUT_SIZE
  -z LATENT_SIZE, --latent_size LATENT_SIZE
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -e EPOCHS, --epochs EPOCHS
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
  -w N_WORKERS, --n_workers N_WORKERS
  -r N_SAMPLES, --n_samples N_SAMPLES
```

## References

* Resnet Paper: [Arxiv](https://arxiv.org/pdf/1512.03385.pdf)
* Triplet Loss Paper: [Arxiv](https://arxiv.org/pdf/1503.03832.pdf)
* TripletTorch Helper Module: [Github](https://github.com/TowardHumanizedInteraction/TripletTorch)

## Todo ( For the students )

**Deadline Decembre 13th 2019 at 12pm**

The students are asked to complete the following tasks:
* Fork the Project
* Improve the model by playing with Hyperparameters and by changing the Architecture ( may not use resnet )
* JIT compile the model ( see [Documentation](https://pytorch.org/docs/stable/jit.html#torch.jit.trace) )
* Add script to generate Centroids and Thesholds using few face images from one person
* Generate those for each of the student included in the dataset
* Add inference script in order to use the final model
* Change README.md in order to include the student choices explained and a table containing the Centroids and Thesholds for each student of the dataset with a vizualisation ( See the one above )
* Send the github link by mail

## My work

You can check the file ia.pynb.


First, I trained the model using the original script provided.

I've obtained these plots after ten epochs:

### Epoch 1:
![vizualisation_0](tripletface/model/vizualisation_0.png)

### Epoch 2:
![vizualisation_1](tripletface/model/vizualisation_1.png)

### Epoch 3:
![vizualisation_2](tripletface/model/vizualisation_2.png)

### Epoch 4:
![vizualisation_3](tripletface/model/vizualisation_3.png)

### Epoch 5:
![vizualisation_4](tripletface/model/vizualisation_4.png)

### Epoch 6:
![vizualisation_5](tripletface/model/vizualisation_5.png)

### Epoch 7:
![vizualisation_6](tripletface/model/vizualisation_6.png)

### Epoch 8:
![vizualisation_7](tripletface/model/vizualisation_7.png)

### Epoch 9:
![vizualisation_8](tripletface/model/vizualisation_8.png)

### Epoch 10:
![vizualisation_9](tripletface/model/vizualisation_9.png)

Then, to improve the model, I decided to use a newer version of Resnet.

According to this [link](https://pytorch.org/docs/stable/torchvision/models.html), there are several version of Resnet which are more effective than Resnet18. So I chose to change the model to a Resnet152. I thought it would provide higher performance and reduce the error.

The only drawback of this version is that it is way heavier than Resnet18 (224Mo), so the training takes more time. And I couldn't upload the model.pt from trainings because it was too heavy.

I've obtained a train_loss=0.000286 and these plots after three epochs:

### Epoch 1:
![vizualisation_0](tripletface/model2/vizualisation_0.png)

### Epoch 2:
![vizualisation_1](tripletface/model2/vizualisation_1.png)

### Epoch 3:
![vizualisation_2](tripletface/model2/vizualisation_2.png)
