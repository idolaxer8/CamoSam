# CamoSam

This repository contains the code and resources for our project on enhancing the SAM model's capabilities for 
camouflaged object detection (COD) tasks by integrating a custom prompt encoder built from BGNet components.

## Model Dependencies

1. Ensure you have the following pre-trained models:
SAM ViT-H: To install the SAM ViT-H model, follow the instructions provided in the SAM repository - https://github.com/facebookresearch/segment-anything .
Res2Net Backbone: Download the res2net50_v1b_26w_4s-3cf99910.pth file from [the official Res2Net repository] (https://github.com/Res2Net/Res2Net-PretrainedModels).

2. downloading testing dataset and move it into ./data/TestDataset/, which can be found in this link https://drive.google.com/file/d/1SLRB5Wg1Hdy7CQ74s3mTQ3ChhjFRSFdZ/view

3. downloading training dataset and move it into ./data/TrainDataset/, which can be found in this  link https://drive.google.com/file/d/1Kifp7I0n9dlWKXXNIbN7kgyokoRY4Yz7/view


## Usage
1.Clone the Repository   ```git clone https://github.com/idolaxer8/CamoSam.git```

2.To train the model, run  train.py

3.Download Pre-trained Models:
Ensure the res2net50_v1b_26w_4s-3cf99910.pth file is in the models directory and install SAM ViT-H as instructed.


## Acknowledgements
This work builds on the SAM model from Facebook Research and the BGNet components for COD. We thank the authors of these works for their contributions to the field.

For more information, please refer to the original papers and repositories:

SAM: Segment Anything
AutoSAM
BGNet
Feel free to raise an issue or contribute to this repository if you have improvements or suggestions.





