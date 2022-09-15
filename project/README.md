# Parsi-IO

### Farsi writing assistant
+ Find grammar mistakes
+ Find spelling mistakes
+ Suggest equivalent words
+ Continuation of the sentence

![](name-of-giphy.gif)

#### Find grammar mistakes
In Farsi, one of the most common mistakes is the incompatibility of the subject with the verb. In order to handle this problem, we developed a module that can return the correct form of the verb that is compatible with the subject of the sentence. In this particular usage, as output, we needed a pair consisting of the primary and corrected word; But it can easily be converted to other formats too.


##### Example
```python
from __future__ import unicode_literals
from hazm import *


vocab = ['پرنده', 'دانشجو', 'درختان']
sentence = 'پرندگان دسته دسته کوچ می کنم.'
sub_verb_matching = Sub_Verb_Matching(vocab)
result = sub_verb_matching.check_matching(sentence)
result

```
##### Output
```python
['می کنم', ''می کنند]

```

#### Find spelling mistakes
Another type of mistakes is spelling errors which includes a wide range of written errors in Farsi. In order to correct such mistakes we came up with pre-trained Bert model whihc has remarkable results. After utilizing several datasets as fine-tune data, we concluded that they're not great as its original model. Ultimately we utilized pre-train Bert model in addition to several techniques (such as levenshtein distance, etc.) to get final result.


##### Example

```python
input = "بسیاری از مباحث علوم غیرطبیعی با استفاده از فیریک دنیای مادی ابل توجیح نیست و برای یادگیری باید به فلسفه‌های خاصی رجو کرد."
result = find_possible_mistakes(input)
result

```

##### Output
```python
'بسیاری از مباحث علوم غیرطبیعی با استفاده از فیزیک دنیای مادی قابل توجیه نیست و برای یادگیری باید به فلسفه‌های خاصی رجوع کرد.'

```

#### Suggest equivalent words
For suitable words in the sentence, it suggests equivalents according to the meaning of the sentence.


##### Example
```python
synWords = SynWords()

text = '.مادر مهربان تمام شب از فرزندش مراقبت کرد'
word = 'مهربان'

syns = synWords.find_equivalent_words(text,word)
syns
```

##### Output
```python
['مشفق','دلسوز']
```


#### Continuation of the sentence
For helping to write, It suggests next word.

##### Example
```python
next_word = NextWord()

seed_text = 'مجلس'
predicted, _ = next_word.predict(seed_text)
predicted

```

##### Output
```python
['خبرگان']
```


###### Contributors
+ Mobina Pournemat
+ Peyman Naseri
+ Narges Javid
+ Zeynab Ehyaee
+ Zahra Khoramnejad


##### And last but not the least, you can see web demo below...

<div align='center'>
    <div><img src='web_demo.gif'/></div>
</div>

