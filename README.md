# BookGenreClassifier-CLIP-Backbone
2023 인공지능경진대회@JBNU, Grand Prize(전북대학교 총장상)

## Model Architecture
![모델 이미지](assets/model_architecture.png) 

## Prerequisties
You can install the required packages with a conda environment by typing the following command in your terminal:
```bash
conda create -n CLIP_KA python=3.9
conda activate CLIP_KA
pip install -r requirements.txt
```
## Usage
### Train our model from scratch
You can train the model with the best hyperparameters for each dataset by typing the following command in your terminal:
```python
python ./src/main.py --learning_rate 1e-5 \
                     --batch_size 20 \
                     --input_dim 768 \
                     --output_dim 24 \
                     --dropout 0.8 \
                     --step_size 5 \
                     --device 'cuda:0'
```
