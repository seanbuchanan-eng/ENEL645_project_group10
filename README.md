# ENEL645_project_group10
Solution for image classification of nutrient deficiencies in winter wheat and winter rye for CVPPA 2023 competition.

# Data Processing
Download the competition dataset from [https://github.com/jh-yi/DND-Diko-WWWR](https://github.com/jh-yi/DND-Diko-WWWR),
update the paths in `process_data.py`, and run.

# Training
Update the paths in the model python files and then run.

- EfficentNet V2 M: `enet.py`
- ResNet50: `resnet.py` (need to download pre-training weights from [https://github.com/jingwu6/Extended-Agriculture-Vision-Dataset](https://github.com/jingwu6/Extended-Agriculture-Vision-Dataset))
- Swin V2 S: `swin.py`
