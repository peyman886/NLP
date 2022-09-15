# import mxnet as mx
# from bert_embedding import BertEmbedding
# import fasttext.util
# import sklearn

from __future__ import unicode_literals

import re
import numpy as np
import pandas as pd
import json
import copy
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModelForMaskedLM
from hazm import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import os
from django.conf import settings
import Levenshtein
from torch import where, topk
from torch.nn import functional as F


dictionary_words = []

def get_vocabs():
    file1 = open('big.txt', 'r', encoding="utf8")
    lines = file1.readlines()

    vocabs = set()

    for line in lines:
        if line.strip() != '':
            vocabs.add(line.strip())

    return vocabs


class SynWords:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
        self.model = AutoModelForMaskedLM.from_pretrained("HooshvareLab/bert-fa-base-uncased")
        self.syn_dict = {}
        self.load_syn_words()

    def load_syn_words(self):
        file1 = open('Farhang_Motaradef-Motazad.txt', 'r', encoding="utf8")
        lines = file1.readlines()

        for line in lines:
            word, s = line.split('&')[0].split(':')
            s = re.sub('\d', '', s)
            s = re.sub('\n', '', s)
            s = re.sub('\u200c', '', s)
            syns = s.split('،')
            self.syn_dict[word.strip()] = syns
          

    def find_equivalent_words(self, text,word ,topN=2):
        output = []

        encoded_input = self.tokenizer(text, return_tensors='pt')
        tokenized_text = self.tokenizer.tokenize(text)

        if self.tokenizer.tokenize(word)[0] not in tokenized_text:
            output = 'معادلی پیدا نشد'
            return output

        word_index = tokenized_text.index(self.tokenizer.tokenize(word)[0])
        model_output = self.model(**encoded_input)
        word_embedding1 = model_output[0][0][word_index]

        if word not in self.syn_dict:
            output = 'معادلی پیدا نشد'
            return output

        syns = self.syn_dict[word]

        similarities = []
        for syn in syns:
            syn = syn.strip()
            text2 = text.replace(word, syn)

            encoded_input = self.tokenizer(text2, return_tensors='pt')

            tokenized_text = self.tokenizer.tokenize(text2)

            if syn == '':
                continue


            if self.tokenizer.tokenize(syn)[0] not in tokenized_text:
                continue

            if syn in tokenized_text:
                word_index = tokenized_text.index(self.tokenizer.tokenize(syn)[0])

                model_output = self.model(**encoded_input)

                word_embedding = model_output[0][0][word_index]
                
                cos_dist = float(1 - cosine(word_embedding.detach().numpy(), word_embedding1.detach().numpy()))
                print(syn,cos_dist)
                similarities.append([syn, cos_dist])

        for _ in range(topN):
            i = np.argmax(np.array(similarities)[:, 1].astype(float))
            syn_i = similarities.pop(i)[0]
            output.append(syn_i)

        return output


class SubVerbMatching:

    def __init__(self, vocab):
        self.vocab = vocab

        with open('mokasar.txt', encoding='utf-8') as file:
            mokasar = [line.rstrip() for line in file]
        self.mokasar_verbs = mokasar

        with open('verb_to_id.json', encoding='utf-8') as f:
            verb_to_id = json.load(f)
        self.verb_to_id = verb_to_id

        with open('id_to_verb.json', encoding='utf-8') as f:
            id_to_verb = json.load(f)
        self.id_to_verb = id_to_verb



    def get_subject_shakhs(self, subject):
        shakhs = 2
        plural_signs = ['ها', 'ان', 'ات', 'ون', 'ین']
        singular_id = ['م', 'ت', 'ش']
        singular_idy = ['هایم', 'هایت', 'هایش']
        plural_id = ['مان', 'تان', 'شان']
        plural_idy = ['هایمان', 'هایتان', 'هایشان']
        pronouns = ['من', 'تو', 'او', 'ما', 'شما', 'آن ها']
        subject = subject.replace('\u200c', '')
        if subject in pronouns:
            shakhs = pronouns.index(subject)
        elif subject[-2:] in plural_signs and subject[:-2] in self.vocab:
            shakhs = 5
        elif subject[-4:] in singular_idy and subject[:-4] in self.vocab:
            shakhs = 5
        elif subject[-1] in singular_id and subject[:-1] in self.vocab:
            shakhs = 2
        elif subject[-4:] in plural_idy and subject[:-4] in self.vocab:
            shakhs = 5
        elif subject[-3:] in plural_id and subject[:-3] in self.vocab:
            shakhs = 5
        elif subject in self.mokasar_verbs:
            shakhs = 5
        elif subject[-3:] == 'گان' and subject[:-3] + 'ه' in self.vocab:
            shakhs = 5
        elif subject[-3:] == 'های' and subject[:-3] in self.vocab:
            shakhs = 5
        elif subject[-3:] == 'یان' and subject[:-3] in self.vocab:
            shakhs = 5
        return shakhs



    def match_verb_with_subject(self, subject, verb):
        id = self.verb_to_id[verb]
        verbs = self.id_to_verb[id]
        shakhs_verb = verbs.index(verb)
        shakhs_sub = self.get_subject_shakhs(subject)
        if shakhs_sub == shakhs_verb:
            return verb
        correct_verb = verbs[shakhs_sub]
        return correct_verb



    def check_matching(self, sentence):
        normalizer = Normalizer()
        tagger = POSTagger(model='resources/postagger.model')
        lemmatizer = Lemmatizer()
        parser = DependencyParser(tagger=tagger, lemmatizer=lemmatizer)
        normalized = normalizer.normalize(sentence)
        words = word_tokenize(normalized)
        tree = parser.parse(words)

        subject, verb = '', ''
        for i in range(len(tree.nodes)):
            node = tree.get_by_address(i)
            if node['rel'] == 'SBJ':
                subject = node['word']
            elif node['rel'] == 'ROOT':
                verb = node['word']
        result = [verb.replace('_',' '), verb]
        try:
            if subject != '':
                verb = verb.replace('\u200c', ' ')
                corrected_verb = self.match_verb_with_subject(subject, verb)
                corrected_verb = normalizer.normalize(corrected_verb)
                result[1] = corrected_verb
            return result
        except:
            return result


class NextWord:
    """
    A class used to predict the next word(s)

    Attributes
    ----------
    next_words : int
        The number of words to predict
    model : 
        A model of type tensorflow.keras.models 
    normalizer: 
        A Hazm normalizer used to normalize seed text
    tokenizer:
        A tokenizer used in trainig phase and also used to tokenize seed text

    Methods
    -------
    predict(seed_text)
        Predicts next_words in given seed_text and returns a string
    """
    def __init__(self, next_words = 1):
        """
        Preparation steps including loading model and defining tokenizer by using train data

        """
        self.next_words = next_words
        self.normalizer = Normalizer(persian_style=False, persian_numbers=False, remove_diacritics=False)

        data = pd.read_csv(os.path.join(settings.PROJECT_ROOT, 'resources/raw_data_for_next_word.csv'))
        data['title'] = data['title'].apply(lambda x: x.replace(u'\xa0',u' '))
        data['title'] = data['title'].apply(lambda x: x.replace('\u200a',' '))

        self.tokenizer = Tokenizer(oov_token='<oov>')
        self.tokenizer.fit_on_texts(data['title'])
        input_sequences = []
        for line in data['title']:
            token_list = self.tokenizer.texts_to_sequences([line])[0]

            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)

        self.max_sequence_len = max([len(x) for x in input_sequences])
        self.model = load_model(os.path.join(settings.PROJECT_ROOT, 'resources/model.h5'))
    
    def predict(self, seed_text):
        """
        Predicts next_words in given seed_text and returns a string,
        it also normalizes the seed text 

        Parameters
        ----------
        seed_text : str
            The text to used to make prediction

        Returns
        -------
        A string containing seed text and predicted next word(s)
        
        """
        normalized_text = self.normalizer.normalize(seed_text)
        seed_text = copy.deepcopy(normalized_text)
        for _ in range(self.next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_len-1, padding='pre')
            predict_x = self.model.predict(token_list, verbose=0) 
            predicted = np.argmax(predict_x,axis=1)

            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
        return seed_text, normalized_text


class SpellChecker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
        self.model = AutoModelForMaskedLM.from_pretrained("HooshvareLab/bert-fa-base-uncased")
    
    def find_possible_mistakes(self, inp, top_k=10000):
        tokens = word_tokenize(inp) # tokenize input
        mistakes = []
        for i, token in enumerate(tokens):
            token = token.replace("\u200C", "")

            #mask ith word in input
            text = " ".join(tokens[:i]) + self.tokenizer.mask_token + " ".join(tokens[i+1:])

            # embedding text 
            input = self.tokenizer.encode_plus(text, return_tensors = "pt")
            mask_index = where(input["input_ids"][0] == self.tokenizer.mask_token_id)

            logits = self.model(**input)
            logits = logits.logits
            softmax = F.softmax(logits, dim = -1)
            mask_word = softmax[0, mask_index, :]
            tops = topk(mask_word, top_k, dim = 1)[1][0]

            
            least_dist = float("inf")
            corrected_word = token
            
            for w in tops:
                word = self.tokenizer.decode([w])
                dist = Levenshtein.distance(token, word)
                if dist < least_dist:
                    corrected_word = word
                    least_dist = dist

            if token != corrected_word:
                for reg in re.finditer(token, inp):
                    s, e = reg.start(), reg.end()
                mistakes.append({"raw": token, "corrected": corrected_word, "span": [s, e]})
        return mistakes











# def pre():
#     global dictionary_words
#     global model_skipgram
#     global refine_data
#     # global prep
#     #
#     # prep = True
#
#     # tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")
#     # model = AutoModelForMaskedLM.from_pretrained("HooshvareLab/bert-fa-base-uncased")
#     #
#     #
#     # SKIPGRAM_MODEL_FILE_ID = '1wPnMG9_GNUVdSgbznQziQc5nMWI3QKNz'
#     # CBOW_MODEL_FILE_ID = '1cQP10CGV6kAwmRuESJ5RTsgHq5TveXwV'
#
#     # gdown.download(id=SKIPGRAM_MODEL_FILE_ID, output="farsi-dedup-skipgram.bin", quiet=False) todo
#
#     # model_skipgram = fasttext.load_model('farsi-dedup-skipgram.bin') todo
#
#     excel_data = pd.read_excel('dictionary.xlsx', header=None)
#     data = pd.DataFrame(excel_data)
#     data.rename(columns={0: 'word', 1: 'synset'}, inplace=True)
#
#     pattern1 = r' *1 *'
#     pattern = r' *[0-9]+ *'
#     data2 = data.dropna()
#     refine_data = pd.DataFrame(columns=['word', 'syn_list'])
#     for index, row in data2.iterrows():
#         syns = re.sub(pattern, '،', re.sub(pattern1, '', row['synset']).strip()).split('&')[0].replace('\u200c', ' ')
#         syn_list = [s.strip() for s in syns.split('،')]
#
#         try:
#             refine_data = refine_data.append({'word': row['word'], 'syn_list': syn_list}, ignore_index=True)
#         except RuntimeWarning:
#             pass
#
#         # refine_data = pd.concat([refine_data,pd.DataFrame.from_records({'word': row['word'], 'syn_list': syn_list})])
#
#     dictionary_words = list(pd.Series(refine_data['word']))
#     dictionary_words = [w.strip() for w in dictionary_words]
#
#
# def find_most_relevant_syn(index, word, sent_word_list, topN):
#     # src_sent_vec = model_skipgram.get_sentence_vector(' '.join(sent_word_list))
#     syn_list = refine_data.loc[dictionary_words.index(word)]['syn_list']
#
#     similarity = []
#
#     # for i in range(len(syn_list)):
#     #     sent_temp = sent_word_list[:]
#     #     sent_temp[index] = syn_list[i]
#     #     similarity.append(
#     #         1 - spatial.distance.cosine(src_sent_vec, model_skipgram.get_sentence_vector(' '.join(sent_temp))))
#
#     # topN_ind = sorted(range(len(similarity)), key=lambda i: similarity[i])[-topN:]
#     # topN_ind.reverse()
#     # topN_syn = []
#
#     # for i in range(topN):
#     #     if syn_list[topN_ind[i]] != '':
#     #         topN_syn.append(syn_list[topN_ind[i]])
#
#     # return topN_syn
#
#     return syn_list
#
#
# def find_equivalent_words(text, topN=1):
#     # print(prep)
#     #
#     # if not prep:
#     #     pre()
#
#     sent_word_list = word_tokenize(text)  # use hazm word tokenizer
#     equivalent_words_list = []
#     ind = 0
#
#     for word in sent_word_list:
#         if word in dictionary_words:
#             equivalent_words_list.append(find_most_relevant_syn(ind, word, sent_word_list, topN))
#         else:
#             equivalent_words_list.append([])
#         ind += 1
#
#     print(equivalent_words_list)
#
#     return equivalent_words_list


def equ_str(equ):
    mstr = 'معادل: '

    if len(equ) > 0:
        mstr += str(equ.pop())
    else:
        mstr += '-'

    while len(equ) > 0:
        mstr = mstr + ', ' + str(equ.pop())

    return mstr
