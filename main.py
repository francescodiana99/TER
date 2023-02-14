#%%
import docx
from docx import Document
import srsly
from doc import *
import random
import spacy
from spacy.tokens import DocBin
from spacy.scorer import Scorer
from spacy.training.example import Example
import re



document = docx.Document("./data/train1.docx")
document_2 = docx.Document("./data/train2.docx")


random.seed(3)

remove_empty_paragraphs(doc=document)
remove_empty_paragraphs(doc=document_2)

data_1 = preprocessing(doc=document)
data_2 = preprocessing(doc=document_2)

# I remove the first 3 elements of the  data of doc 1, because they are part of the legend
data_1.pop(0)
data_1.pop(0)
data_1.pop(0)

data_2.pop(0)
data_2.pop(0)
data_2.pop(0)

# I merge the two lists
data = data_1 + data_2
#print(len(data))

# shuffling data
random.shuffle(data)

# separate data in training and evaluation
train_index = round(len(data)*0.8)
train_data = data[:train_index]
eval_data = data[:train_index]

#%%


# I check the number of annotation for each label
NLD = 0
DRUG = 0
ADR = 0
for _, annot in data:
    nld = re.findall("NLD")
    drug = re.findall("DRUG")
    re.findall("ADR")
    NLD = NLD + len(nld)
    DRUG = DRUG + len(drug)
    ADR = ADR + len("ADR")

print(NLD)
print(DRUG)
print(ADR)




#%%
# ================================== DOCBIN CREATION =================================================

db = DocBin()
nlp = spacy.blank("fr")
for text, annotations in train_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations:
        # can't correctly select Amoxicilline (7th example)
        span = doc.char_span(start, end, label=label)
        #print(span)
        ents.append(span)

    doc.ents = ents
    db.add(doc)
db.to_disk("./train.spacy")

for text, annotations in eval_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations:
        # can't correctly select Amoxicilline (7th example)
        span = doc.char_span(start, end, label=label)
        #print(span)
        ents.append(span)

    doc.ents = ents
    db.add(doc)
db.to_disk("./eval.spacy")



# %%
# ===================================== PIPELINE SETUP =========================================


#nlp = spacy.blank("fr")
# this should be ran after runnin python -m spacy train config.cfg --paths.train ./train.spacy --paths.dev ./eval.spacy --gpu-id 0
nlp_ner = spacy.load("./models/model-best")

# adding dictionary to the pipeline
patterns = srsly.read_jsonl("pattern.jsonl")
ruler = nlp_ner.add_pipe("span_ruler", first=True)
ruler.add_patterns(patterns)

#%%
print(nlp_ner._components)
# %%

# ======================= TEST ========================================
test = nlp_ner("Patiente initialement admise à la clinique pour un épisode dépressif sévère. L'évolution s'est montrée peu favorable lors de cette hospitalisation. La patiente est donc transférée dans le service. La patiente s'est rapidement dégradée sur le plan somatique avec hyperthermie et signes neurologiques, notamment suite à un surdosage en augmentin ayant entraîné une insuffisance rénale aigue sur son insuffisance rénale chronique. Dans ce contexte, la patiente est transférée en service. Hospitalisée du 10/06/2021 au 23/06/2021 pour prise en charge de troubles de la vigilance fébriles. ")
colors = {"DRUG": "#F67DE3", "ADR": "#7DF6D9", "NLD":"#FFFFFF"}
options = {"colors": colors}

spacy.displacy.render(test, style="ent", options= options, jupyter=True)

# %%