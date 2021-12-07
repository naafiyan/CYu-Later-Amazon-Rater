# CYu-Later-Amazon-Rater

# Command Line Args
```
python3 file_name [--num_examples NUM_EXAMPLES] [--batch_size BATCH_SIZE] [--epochs NUM_EPOCHS] [--lr LEARNING_RATE]
```

Args in [] are optional
```
Default values:
NUM_EXAMPLES = 10000
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
```

Note: file_name should not include .json.gz file extension

## File Structure
```
project/
├─ code/
│  ├─ main.py
│  ├─ preprocess.py
│  ├─ model.py
data/
├─ <data_name>.json.gz
```
