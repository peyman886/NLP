from __future__ import unicode_literals
from hazm import *
import json



class Sub_Verb_Matching:

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
        elif subject[-2:] in plural_signs and subject[:-2] in vocab:
            shakhs = 5
        elif subject[-4:] in singular_idy and subject[:-4] in vocab:
            shakhs = 5
        elif subject[-1] in singular_id and subject[:-1] in vocab:
            shakhs = 2
        elif subject[-4:] in plural_idy and subject[:-4] in vocab:
            shakhs = 5
        elif subject[-3:] in plural_id and subject[:-3] in vocab:
            shakhs = 5
        elif subject in self.mokasar_verbs:
            shakhs = 5
        elif subject[-3:] == 'گان' and subject[:-3] + 'ه' in vocab:
            shakhs = 5
        elif subject[-3:] == 'های' and subject[:-3] in vocab:
            shakhs = 5
        elif subject[-3:] == 'یان' and subject[:-3] in vocab:
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
        result = [verb, verb]
        try:
            if subject != '':
                verb = verb.replace('\u200c', ' ')
                corrected_verb = self.match_verb_with_subject(subject, verb)
                corrected_verb = normalizer.normalize(corrected_verb)
                result[1] = corrected_verb
            return result
        except:
            return result




def main(sentence, vocab):
    try:
        sub_verb_matching = Sub_Verb_Matching(vocab)
        result = sub_verb_matching.check_matching(sentence)
        return result
    except:
        pass



# ============================== test case =============================== #

# vocab = ['درخت', 'درختان', 'شکوفه', 'ارتباط', 'شاکر', 'دختر', 'پرنده', 'ستاره', 'دانشجو', 'دست', 'دستان', 'اطلاعات', 'اطلاع', 'لباس']
# sent = 'دانشجویان لباس می شست.'
# print(main(sent, vocab))

# Output: ['می\u200cشست', 'می\u200cشستند']  