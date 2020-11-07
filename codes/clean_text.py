import spacy
from scispacy.abbreviation import AbbreviationDetector
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
stop_words = set(stopwords.words('english'))

nlp1 = spacy.load("en_core_web_sm")
abbreviation_pipe1 = AbbreviationDetector(nlp1)
nlp1.add_pipe(abbreviation_pipe1)
nlp2 = spacy.load("en_core_sci_sm")
abbreviation_pipe2 = AbbreviationDetector(nlp2)
nlp2.add_pipe(abbreviation_pipe2)

def acronym(text):
    doc = nlp1(text)
    acro = doc._.abbreviations
    return acro
def bio_term(text):
    doc = nlp2(text)
    list_ = [str(i) for i in list(doc.ents)]
    return list_
def remove_words(text):
    text1 = re.sub(r"[^a-zA-Z]", " ", text)
    tagged_sent = pos_tag(text1.split())
    noun = [word for word,pos in tagged_sent if pos == 'NNP']
    acro = acronym(text)
    boi_ = list(bio_term(text))
    Exemptions = ['pH']
    final_list = list(set().union(noun, acro,Exemptions))
    return final_list
def remove_number(text):
    clean_txt = re.sub(r"(^[\d.-][)])|(^[\d.-]+\s*)|(^[a-z0-9A-Z][[{.:(+*)])|(^[[{.: (+*)][a-z0-9A-Z][[{.:(+*)])|(^[C|I|L|V|X]{0,6}[.-]+\s*)", "", text)
    clean_txt1 = re.sub(r"[^a-zA-Z]", " ",clean_txt)
    return clean_txt1 
def extra_stopwords(text):
    tag_2 = pos_tag(word_tokenize(text),tagset='universal')
    new_stop = [word.lower() for word,pos in tag_2 if pos in ['ADP','CONJ','DET']]
    stop_words1 = list(stop_words) + new_stop if new_stop is not None else stop_words
    return stop_words1
def proper_noun(text):
    text1 = re.sub(r"[^a-zA-Z]", " ", text)
    tag_1 = pos_tag(word_tokenize(text1))
    noun = [word for word,pos in tag_1 if pos == 'NNP']
    stop_words1 = extra_stopwords(text1)
    words = [i for i in word_tokenize(text1) if i.lower() not in stop_words1]
    pre_ = len(noun)/len(words) if len(words)>0 else 0
    if (len(noun)==len(words))&(len(words)>0):
        value = 1
    else:
        value = 0
    return [value,noun,pre_]
def clean_text(text,arg):
    clean_txt1 = remove_number(text)
    stop_words1 = extra_stopwords(clean_txt1)
    tokens = word_tokenize(clean_txt1)
    if arg:
        remove_ = [str(i).lower() for i in remove_words(clean_txt1) if i]
        stop_words2 = stop_words1 + remove_
    else:
        stop_words2 = stop_words1
    words = [i for i in tokens if i.lower() not in stop_words2]
    return words

# sentence extraction function
def sent_ext(text):
    try:
        doc = sent_tokenize(str(text))
        return doc[0],len(doc)
    except:
        return ''
# caps after colon
def caps_after_colon(text):
    list1 = text.split(':')
    if len(list1)>1:
        temp = list1[1].split()[0][0].isupper() if len(list1[1].split())>0 else None
        if temp:
            return 1
        else:
            return 0
    else:
        return 0
# text before colon and text after colon
def word_before_colon(text):
    word = text[:300].split(':')
    len_ = len(re.findall(r'\w+', word[0])) if len(word)>1 else 0
    return len_
def word_after_colon(text):
    word = text[:300].split(':')
    len_ = len(re.findall(r'\w+', word[1]))  if len(word)>1 else 0
    return len_
def pre_caps(text):
    tokens = clean_text(text,0)
    word_ = [i.strip() for i in tokens if i.strip()[0].isupper()]
    pre_ = (len(word_)/len(tokens)) if len(tokens)>0 else 0
    if  (pre_>= 0.8):
        return 1
    else:
        return 0
def word_present(text):
    l = [item.lower() for item in clean_text(text,0)]
    l1 = any([i for i in ['table','tables','fig','figure','figures','exhibit'] if i in l])
    return int(l1)
def math_expression(mystr):
    l1 = re.split("([+*=⨉×↔≡^])", mystr.replace(" ", "")) # remove x after put 
    l2 = any([i for i in ['+','*','⨉','×', '=','↔','≡','^'] if i in l1])
    return int(l2)
def number_only(text):
    regex = re.compile('[^a-zA-Z0-9]')
    l1 = regex.sub('', text)
    if l1.isnumeric():
        return 1
    else:
        return 0
def roman_only(text):
    regex = re.compile('[C|I|L|V|X]{0,6}[.-]+\s*')
    l1 = regex.sub('', text)
    if l1.isnumeric():
        return 1
    else:
        return 0
def DNA_seq(text):
    regex = re.compile('[^a-zA-Z]')
    l1 = regex.sub('', text)
    l2 = re.search('CC(G|C|A|T)|AG(CT|C|U|T)|(TC(G|A|GA|U))|(N[^P][ST][^P])|(CAG|CAA){1,20}', l1)
    if l2:
        return 1
    else:
        return 0
def no_after_word(text):
    list_ = ['table','tables','fig','figure','figures']
    regex1 = re.compile('[^a-zA-Z0-9]')
    regex2 = re.compile("(table(.*)\s{0,2}\d)|(tables(.*)\s{0,2}\d)|(fig(.*)\s{0,2}\d)|(figure(.*)\s{0,2}\d)|(figures(.*)\s{0,2}\d)")
    l1 = regex1.sub(' ', text)
    try:
        l2 = regex2.search(l1.lower()).group(0)
    except:
        l2 = None
    l3 = l2.strip().split()[-1].isnumeric() if l2 is not None else None
    if l3:
        return 1
    else:
        return 0
def number_hir(text):
    regax = re.compile("^[\d.-]+\s*")
    try:  
        a1=list(match.group() for match in regax.finditer(text))
        a2 = a1[0].strip().split('.')
        a3 = [i for i in a2 if i]
        m = re.match(r'(\d+)[^\d]+(\d+)', text).group(2) if len(a3)>1 else '1'
        if m =='0':
            return 1
        else:
            return len(a3)
    except:
        return 0
def year_persent(text):
    regax = re.compile("([1-3][0-9]{3})")
    b1=list(match.group() for match in regax.finditer(text))
    if len(b1)>0:
        return 1
    else:
        return 0
def sign_persent(text):
    regax = re.compile("(^<(.*)>$)|(^{(.*)}$)")
    b1=list(match.group() for match in regax.finditer(text.strip()))
    if len(b1)>0:
        return 1
    else:
        return 0
def startswthalpha(text):
    regax = re.compile(r"(^[a-zA-Z][[{.:(+*)])|(^[[{.: (+*)][a-zA-Z][[{.:(+*)])")
    b1=list(match.group() for match in regax.finditer(text.strip()))
    if len(b1)>0:
        return 1
    else:
        return 0
def startswthnum(text):
    regax = re.compile(r"(^[\d.-][)])|(^[\d.-]+\s*)|(^[0-9][[{.:(+*)])|(^[[{.: (+*)][0-9][[{.:(+*)])")
    b1=list(match.group() for match in regax.finditer(text.strip()))
    if len(b1)>0:
        return 1
    else:
        return 0
def startswthroman(text):
    regax = re.compile(r"(^[I|L|V|X]{1,6}[.])")
    b1=list(match.group() for match in regax.finditer(text.strip()))
    if len(b1)>0:
        return 1
    else:
        return 0
def state_words(text):
    regex = re.compile('(^hypothesis(.*)\s[\d^#:])|(^lemma(.*)\s[\d^#:])|(^theorem(.*)\s[\d^#:])|(^definition(.*)\s[\d^#:])')
    try:
        regex_ = regex.search(text.lower()).group(0)
    except:
        regex_ = 0
    if regex_:
        return 1
    else:
        return 0