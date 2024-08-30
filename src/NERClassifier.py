from transformers import pipeline
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForTokenClassification
import torch
import json



class PERextractor:
    def __init__(self, _model_path, _tokenizer_name, _id2label):
        self.model = AutoModelForTokenClassification.from_pretrained(_model_path) 
        self.tokenizer = AutoTokenizer.from_pretrained(_tokenizer_name)
        self.id2label = _id2label

    def get_NER_tags(self, _text):
        text = self.tokenizer(_text, return_tensors='pt')
        text.pop('token_type_ids')
        logits = self.model(**text).logits

        text = self.tokenizer(_text)
        inds = text.word_ids()
        tokens = self.tokenizer.convert_ids_to_tokens(text["input_ids"])

        words = []
        word = []
        argmax = []
        argmaxs = []
        for i in range(len(inds) - 1):
            if inds[i]!=None:
                if inds[i] != inds[i+1]:
                    argmax.append(torch.argmax(logits[0][i]))
                    argmaxs.append(argmax)
                    word.append(tokens[i])
                    words.append(word)
                    word = []
                    argmax = []
                else:
                    word.append(tokens[i])
                    argmax.append(torch.argmax(logits[0][i]))
        res_words = []
        res_tags = []
        for word in words:
            res_words.append(''.join(word).replace('#', ''))

        for argmax in argmaxs:
            res_tags.append(self.id2label[argmax[0].item()])

        return res_words, res_tags
    
    def singlePredictDict(self, text):
        words, tags = self.get_NER_tags(text)
        positions = []
        for i, word in enumerate(words):
            if tags[i] == 'PER':
                start_position = text.find(word)
                end_position = -1
                if start_position != -1:
                    end_position = start_position + len(word) - 1
                positions.append([word, start_position, end_position])
        dictionary = {'entities': []}
        if len(positions) != 0:
            for elem in positions:
                el_dict = {'value': elem[0],
                        'entity_type': 'PERSON',
                        'span': {'start_index':elem[1], 'end_index':elem[2]},
                        'entity': elem[0],
                        'source_type': 'SLOVNET'}
                dictionary['entities'].append(el_dict)
        return dictionary

    def makePredictions(self, inputJSON, outputJSON='predictions.json'):
        f = open(inputJSON, encoding='UTF-8')
        data = json.load(f)
        dictionary = {'enteties_list': []}
        for text in data['texts']:
            d = self.predictDict(text)
            dictionary['enteties_list'].append(d)

        with open(outputJSON, "w", encoding='UTF-8') as outfile:
            json.dump(dictionary, outfile, ensure_ascii=False, indent=4)
        return dictionary
    
    def predictionsFromDict(self, data):
        dictionary = {'enteties_list': []}
        for text in data['texts']:
            d = self.predictDict(text)
            dictionary['enteties_list'].append(d)
        return dictionary
        


