#%%
import docx
from docx import Document
import srsly
from doc import *
import random
import spacy
from spacy.tokens import DocBin




document = docx.Document("./data/train1.docx")
document_2 = docx.Document("./data/train2.docx")


random.seed(22)

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


def DocBin_to_disk(data:list, path:str):
    """create a DocBin object from a list of annotations and saves it on disk"""

    db = DocBin()
    nlp = spacy.blank("fr")
    for text, annotations in data:
        doc = nlp(text)
        ents = []
        for start, end, label in annotations:
            # can't correctly select Amoxicilline (7th example)
            span = doc.char_span(start, end, label=label)
            #print(span)
            ents.append(span)

        doc.ents = ents
        db.add(doc)
    db.to_disk(path)

#%%
# ================================== DOCBIN CREATION =================================================
DocBin_to_disk(train_data,"./train_2.spacy")
DocBin_to_disk(eval_data, "./eval_2.spacy")

# db = DocBin()
# nlp = spacy.blank("fr")
# for text, annotations in train_data:
#     doc = nlp(text)
#     ents = []
#     for start, end, label in annotations:
#         # can't correctly select Amoxicilline (7th example)
#         span = doc.char_span(start, end, label=label)
#         #print(span)
#         ents.append(span)

#     doc.ents = ents
#     db.add(doc)
# db.to_disk("./train_2.spacy")

# for text, annotations in eval_data:
#     doc = nlp(text)
#     ents = []
#     for start, end, label in annotations:
#         # can't correctly select Amoxicilline (7th example)
#         span = doc.char_span(start, end, label=label)
#         #print(span)
#         ents.append(span)

#     doc.ents = ents
#     db.add(doc)
# db.to_disk("./eval.spacy")



# %%
# ===================================== PIPELINE SETUP =========================================


#nlp = spacy.blank("fr")
# this should be ran after runnin python -m spacy train config.cfg --paths.train ./train.spacy --paths.dev ./eval.spacy --gpu-id 0
nlp_ner = spacy.load("./models/model_2/model-best")

# adding dictionary to the pipeline
patterns = srsly.read_jsonl("pattern.jsonl")
ruler = nlp_ner.add_pipe("span_ruler", before='ner')
ruler.add_patterns(patterns)
nlp_ner.to_disk("./models/pipeline_2")
#%%
print(nlp_ner.pipe_names)

# %%
# ======================= TEST ========================================
test = nlp_ner("11)	Avant son épisode de sclérite, le patient a bénéficié de quelques cures de cortisone, puis d’AINS et depuis l’émergence de la sclérite antérieure de l’œil gauche, introduction d’une corticothérapie à 50 mg avec régression de 10 mg toutes les semaines.")
colors = {"DRUG": "#F67DE3", "ADR": "#7DF6D9", "NLD":"#FFFFFF"}
options = {"colors": colors}

spacy.displacy.render(test, style="ent", options= options, jupyter=True)

# %%