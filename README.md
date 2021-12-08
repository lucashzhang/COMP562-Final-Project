# COMP562-Final-Project

## Main results

### Performance

![Model Table](./readme/Table.png)

## Set up
Upload dataset into `data`. Make sure to maintain two folder subsets for binary classification. Labels are inferred from folder names.
Moidify data paths in `utils/dataloader.py`.

## Evaluation
Download the model in `saved_models/models.txt` into `saved_models`, or train your own.
```
python test.py
```

## Training
Modify the `exp_id` and `log_id` in `train.py`.
```
python train.py
```

### Final Report
[Final Report PDF link](562_Final_Project.pdf)