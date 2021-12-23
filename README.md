# CYu-Later-Amazon-Rater
[Devpost Submission](https://devpost.com/software/cyu-later-amazon-rater)
[Final Report](https://docs.google.com/document/d/1YONKM_1sVW1mKcL3nkIq_mz-WqDJFp57tLytqVlCPVU/edit?usp=sharing)

# Command Line Args
```
python3 file_name [--num_examples NUM_EXAMPLES] [--batch_size BATCH_SIZE] [--epochs NUM_EPOCHS] [--lr LEARNING_RATE] [--sentiment_threshold THRESHOLD]
```

Args in [] are optional
```
Default values:
NUM_EXAMPLES = 10000
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
THRESHOLD = 3.5
```

Note: file_name should not include .json.gz file extension

## File Structure
```
project/
├─ code/
│  ├─ glove.py
│  ├─ main.py
│  ├─ mode.py
│  ├─ preprocess.py
├─ data/
│  ├─ <data set>.json.gz
├─ models/
│  ├─ <data set>.h5
```
