from nerus import load_nerus
import pandas as pd
from tqdm import tqdm

NERUS = 'nerus_lenta.conllu.gz'
docs = load_nerus(NERUS)
doc = next(docs)
data = []
for doc in tqdm(docs):
    sentence = []
    tags = []

    for token in doc.sents[0].tokens:
        sentence.append(token.text)
        if 'PER' in token.tag:
            tags.append('PER')
        else:
            tags.append(token.tag)
    sentence = ' '.join(sentence)
    tags = ' '.join(tags)
    data.append([sentence, tags])

df = pd.DataFrame(data=data, columns=['sentence', 'tags'])
train=df.sample(frac=0.8,random_state=200)
test=df.drop(train.index)
train.to_csv('train.csv', sep=',', index=False)
test.to_csv('test.csv', sep=',', index=False)

