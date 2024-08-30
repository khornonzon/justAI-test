from NERClassifier import PERextractor
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score
from tqdm import tqdm
import numpy as np
import time
id2label = {0: 'B-LOC', 1: 'B-ORG', 2: 'I-LOC', 3:'I-ORG', 4:'O', 5: 'PER'}

class Tester:
    def __init__(self, _path_csv, _path_to_model):
        self.df = pd.read_csv(_path_csv)
        self.pipeline = PERextractor(_path_to_model, "DeepPavlov/rubert-base-cased", id2label)
    

    def get_preds_and_targets(self, sample):
        data = self.df.sample(sample, random_state=43)
        target = data['tags']
        texts = data['sentence']
        target = target.map(lambda x: x.split(' '))
        target_values = target.to_numpy()
        preds = []
        targets = []
        for i, text in enumerate(texts):
            res_text, result = self.pipeline.get_NER_tags(text)
            if len(result) == len(target_values[i]):
                preds.append(result)
                targets.append(target_values[i])
            else:
                continue
        return preds, targets
    
    def recall(self, sample):
        preds, targets = self.get_preds_and_targets(sample)
        recalls = []

        for i in range(len(preds)):
            recall = recall_score(targets[i], preds[i], average='micro')
            recalls.append(recall)
        return np.mean(recalls)

    def precision(self, sample):
        preds, targets = self.get_preds_and_targets(sample)
        precisions = []

        for i in range(len(preds)):
            precision = precision_score(targets[i], preds[i],average='micro')
            precisions.append(precision)
        return np.mean(precisions)
    def f1(self, sample):
        preds, targets = self.get_preds_and_targets(sample)
        f1s = []

        for i in range(len(preds)):
            f1 = f1_score(targets[i], preds[i],average='micro')
            f1s.append(f1)
        return np.mean(f1s)
    
    def summary_metrics(self, sample):
        preds, targets = self.get_preds_and_targets(sample)
        f1s, recalls, precisions = [], [], []
        for i in range(len(preds)):
            f1 = f1_score(targets[i], preds[i],average='macro')
            f1s.append(f1)
            precision = precision_score(targets[i], preds[i],average='macro')
            precisions.append(precision)
            recall = recall_score(targets[i], preds[i], average='macro')
            recalls.append(recall)

        return {'precision': np.mean(precisions),
                'recall': np.mean(recalls),
                'f1': np.mean(f1s)}
    
    def avg_speed(self, sample):
        data = self.df.sample(sample, random_state=43)
        texts = data['sentence']
        vels = []
        for text in texts:
            
            start_time = time.time()
            res_text, result = self.pipeline.get_NER_tags(text)
            end_time = time.time()
            elapsed_time = end_time - start_time
            num_chars = len(text)
            speed_chars_per_sec = num_chars / elapsed_time
            vels.append(speed_chars_per_sec)
        return np.mean(vels)

    



