from django.shortcuts import render
from django.http import HttpResponse
import re
import json
from .utils import equ_str,SynWords,SubVerbMatching,get_vocabs, NextWord, SpellChecker
from django.views.decorators.csrf import csrf_exempt

# for running only once:
synWords = SynWords() 
vocab = get_vocabs()
sub_verb_matching = SubVerbMatching(vocab)
next_word = NextWord()
spell_checker = SpellChecker()


def home(request):
    return render(request, 'grammarly/home.html')


def svmhome(request):
    return render(request, 'grammarly/svmhome.html')


def synhome(request):
    return render(request, 'grammarly/synhome.html')


def grammarhome(request):
    return render(request, 'grammarly/grammarhome.html')


def get_synonym(request):
    if request.method == 'POST':
        text =  request.POST.get('your_text')
        word =  request.POST.get('your_word')

        # if re.match(r'^[\u0600-\u06FF\s0-9]+$', text):
        syns = synWords.find_equivalent_words(text,word)
        return render(request, 'grammarly/synresult.html', {'word': word, 'data' : syns})

    return render(request, 'grammarly/error.html', {'error': 'ورودی نامعتبر می باشد'})


def get_subverb_match(request):
    if request.method == 'POST':
        text =  request.POST.get('your_text')

        # if re.match(r'^[\u0600-\u06FF\s0-9]+$', text):
        result = sub_verb_matching.check_matching(text)
        print('-----------',result)

        idx = text.index(result[0])
        if result[0] == result[1] :
            context = {'data' : [{
                        'phrase': text,
                        'formattedPhrase' : "",
                        'message': "",
                        'typeCode' : 0
                    },], 
                        'numberOfIssues' : 0}
        else:
            data = [{
                'phrase': text[0: idx],
                'formattedPhrase' : "",
                'message': "",
                'typeCode' : 0
            }, 
            {
                'phrase': result[0],
                'formattedPhrase' : result[1],
                'message': "عدم تطابق فعل با شناسه" + " شکل صحیح \n : " + result[1],
                'typeCode' : 1
            }, 
            {
                'phrase': text[idx + len(result[0]):],
                'formattedPhrase' : "",
                'message': "",
                'typeCode' : 0
            }, 
            ]
            context = {'data' : data, 'numberOfIssues' : 1}

        return render(request, 'grammarly/svmresult.html', context)

    return render(request, 'grammarly/error.html', {'error': 'ورودی نامعتبر می باشد'})

@csrf_exempt
def get_next_word(request):
    try:
        seed_text = request.POST.get('word')
        predicted, normalized_text = next_word.predict(seed_text)

        response_data = {}
        response_data['next_word'] = predicted
        response_data['normalized_text'] = normalized_text
    
        return HttpResponse(json.dumps(response_data), content_type="application/json", status=200)
        
    except Exception:
        response_data = {}
        response_data['status'] = 'fail'
        response_data['message'] = 'Some error occurred'

        return HttpResponse(json.dumps(response_data), content_type="application/json", status=200)


def get_grammar_check(request):
    if request.method == 'POST':
        text =  request.POST.get('your_text')

        # if re.match(r'^[\u0600-\u06FF\s0-9]+$', text):
        result = spell_checker.find_possible_mistakes(text)
        if len(result) == 0 :
            context = {'data' : [{
                        'phrase': text,
                        'formattedPhrase' : "",
                        'message': "",
                        'typeCode' : 0
                    },], 
                        'numberOfIssues' : 0}
        else:
            last_idx = 0
            data = []
            for res in result:
                data.extend([
                {
                'phrase': text[last_idx: res["span"][0]],
                'formattedPhrase' : "",
                'message': "",
                'typeCode' : 0
                },
                {
                    'phrase': res["raw"],
                    'formattedPhrase' : res["corrected"],
                    'message': "اشکال نگارشی یا گرامری" + " شکل صحیح \n : " + res["corrected"],
                    'typeCode' : 2
                }])
                last_idx = res["span"][1]
            
            data.append({
                'phrase': text[result[-1]["span"][1] : ],
                'formattedPhrase' : "",
                'message': "",
                'typeCode' : 0
            })
            

            context = {'data' : data, 'numberOfIssues' : len(result)}

        return render(request, 'grammarly/grammarresult.html', context)

    return render(request, 'grammarly/error.html', {'error': 'ورودی نامعتبر می باشد'})



