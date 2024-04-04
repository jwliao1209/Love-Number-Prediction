# Love-Number-Prediction

## Setup
```
virtualenv --python=python3.10 love_pred
source love_pred/bin/activate
pip install -r envs/requirements.txt
```

## Training
```
python train.py --model 2step-lgbm 2step-xgboost 2step-catboost
```
- `model`: `{1step, 2step}-{xgboost, lgbm, catboost}`

## Citation
```
@misc{
    title  = {Love-Number-Prediction},
    author = {Jia-Wei Liao},
    url    = {https://github.com/jwliao1209/Love-Number-Prediction},
    year   = {2024}
}
```

