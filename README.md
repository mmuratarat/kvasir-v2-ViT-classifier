This repo contains the artifacts of a model, using a pre-trained Visual Transformer model and fine-tuning it on a custom dataset.

There are significant benefits to using a pretrained model. It reduces computation costs, your carbon footprint, and allows you to use state-of-the-art models without having to train one from scratch

 🤗 Hugging Face Transformers provides access to thousands of pretrained models for a wide range of tasks. When you use a pretrained model, you train it on a dataset specific to your task. This is known as fine-tuning, an incredibly powerful training technique.

## Dataset

Here, we use Kvasir dataset v2. It consists of images, annotated and verified by medical doctors (experienced endoscopists), including several classes showing anatomical landmarks, phatological findings or endoscopic procedures in the gastrointestinal tract

It is a multi-class dataset consisting of 1,000 images per class with a total of 8,000 images for eight different classes. These classes consist of pathological findings (esophagitis, polyps, ulcerative colitis), anatomical landmarks (z-line, pylorus, cecum), and normal and regular findings (normal colon mucosa, stool), and polyp removal cases (dyed and lifted polyps, dyed resection margins)

The dataset can be download from [here](https://datasets.simula.no/kvasir/) which weights around ~2.3 GB. and is free for research and educational purposes only. 

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/kvasir_v2_examples.png?raw=true)

## Model

The [Hugging Face transformers](https://huggingface.co/docs/transformers/index) package is a very popular Python library which provides access to the HuggingFace Hub where we can find a lot of pretrained models and pipelines for a variety of tasks in domains such as Natural Language Processing (NLP), Computer Vision (CV) or Automatic Speech Recognition (ASR).

A Vision Transformer-based model is used in this experiment. Vision Transformer (ViT) was introduced in June 2021 by a team of researchers at Google Brain (https://arxiv.org/abs/2010.11929). 

The Vision Transformer (ViT) model was proposed in "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. It’s the first paper that successfully trains a Transformer encoder on ImageNet, attaining very good results compared to familiar convolutional architectures.

The Vision Transformer (ViT) is a transformer encoder model (BERT-like) pretrained on a large collection of images in a supervised fashion, namely ImageNet-21k, at a resolution of 224x224 pixels. Next, the model was fine-tuned on ImageNet (also referred to as ILSVRC2012), a dataset comprising 1 million images and 1,000 classes, also at resolution 224x224.

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/Screenshot%202023-10-04%20at%205.17.43%20PM.png?raw=true)

In this case, we'll be using [the google/vit-base-patch16-224-in21k model](https://huggingface.co/google/vit-base-patch16-224-in21k) from Hugging Face.

## Training hyperparameters

The following hyperparameters were used during training:

* learning_rate: 2e-5
* lr_scheduler_type: linear
* warmup_steps = 500
* weight_decay = 0.01
* warmup_ratio=0.0
* train_batch_size: 16
* eval_batch_size: 32
* seed: 42
* num_epochs: 5
* optimizer: Adam
* adam_beta1=0.9,
* adam_beta2=0.999,
* adam_epsilon=1e-08,

## Evaluation Metrics

We have used usual evaluation metrics for this image classification task, that include weighted average precision, F1 score and recall, and overall accuracy. In multi-class classification problems, the weighted average method adjusts for class imbalance by assigning a weight proportional to the number of instances in each class.

## Training results

| **Epoch**     | **Training Loss**     | **Validation Loss**     | **Accuracy**     |  **F1**     | **Precision**     | **Recall**     |
|:---------:    |:-----------------:    |:-------------------:    |:------------:    |:-------:    |:-------------:    |:----------:    |
|     1         |       1.4341          |       0.62736           |    0.89417       | 0.89285     |    0.90208        |   0.89417      |
|     2         |       0.4203          |        0.3135           |    0.92917       |  0.929      |    0.93058        |   0.92917      |
|     3         |       0.2272          |        0.251            |    0.9375        | 0.93745     |     0.938         |   0.9375       |
|     4         |       0.146           |       0.24937           |    0.93833       | 0.93814     |    0.94072        |   0.93833      |
|     5         |       0.1034          |        0.2383           |    0.93917       |  0.9391     |    0.93992        |   0.93917      |

![](https://github.com/mmuratarat/turkish/blob/master/_posts/images/kvasir_vit_model_progress.png?raw=true)

## Training Metrics

    epoch                    =                   5.0
    total_flos               = 2.634869332279296e+18
    train_loss               =   0.46618968290441176
    train_runtime            =            0:01:51.45
    train_samples_per_second =                  5.07
    train_steps_per_second   =                 0.317
    global_step              =                  2125

## Fine-tuned model

The pre-trained model has been pushed to Hugging Face Hub and can be found on https://huggingface.co/mmuratarat/kvasir-v2-classifier.

You can make inferences by either using "Hosted Inference API" on the Hub or locally pulling the model from the Hub.

## How to use

Here is how to use this pre-trained model to classify an image of GI tract:

```python
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import requests

# this is an image from "polyps" class
url = 'https://github.com/mmuratarat/turkish/blob/master/_posts/images/example_polyps_image.jpg?raw=true'
image = Image.open(requests.get(url, stream=True).raw)

model = AutoModelForImageClassification.from_pretrained("mmuratarat/kvasir-v2-classifier")
feature_extractor = AutoFeatureExtractor.from_pretrained("mmuratarat/kvasir-v2-classifier")
inputs = feature_extractor(image, return_tensors="pt")

id2label = {'0': 'dyed-lifted-polyps', 
            '1': 'dyed-resection-margins', 
            '2': 'esophagitis', 
            '3': 'normal-cecum', 
            '4': 'normal-pylorus', 
            '5': 'normal-z-line', 
            '6': 'polyps', 
            '7': 'ulcerative-colitis'}

logits = model(**inputs).logits
predicted_label = logits.argmax(-1).item()
predicted_class = id2label[str(predicted_label)]
predicted_class
```

## Framework versions

* Transformers 4.34.0
* Pytorch 2.0.1+cu118
* Datasets 2.14.5
* Tokenizers 0.14.0
* scikit-learn 1.2.2
* scipy 1.11.3
* numpy 1.23.5
* accelerate 0.23.0
* pandas 1.5.3

## Contact

Please reach out to arat.murat@gmail.com if you have any questions or feedback.

## Source Code

You can find the source code for obtaining this pre-trained model on [mmuratarat/kvasir-v2-ViT-classifier]( https://github.com/mmuratarat/kvasir-v2-ViT-classifier) repository of Github.

Note that `Kvasir_ViT.ipynb` file contains Turkish commentary but the code itself is self-explanatory.

## Citation

In all documents and papers that use or refer to this pre-trained model or report benchmarking results, a reference to this model have to be included.
