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
from spacy.pipeline import EntityRecognizer

#%%
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
nlp = spacy.load('fr_dep_news_trf')

TRAINING_DATA = []
for text, ann in train_data:
    TRAINING_DATA.append((text, {"entities": ann}))

# #%%
# examples = []
# for text, ann in TRAINING_DATA:
#     examples.append(Example.from_dict(nlp.make_doc(text), ann))


# ner = nlp.add_pipe("ner")
# ner.add_label("DRUG")
# ner.add_label("NLD")
# ner.add_label("ADR")

# print(ner.labels)
# #%%


# optimizer = nlp.create_optimizer()
# losses = ner.update(examples)
#%%



# TRAINING_DATA = [("Example text with entity", {"entities": [(0, 14, "ENTITY")]})]

# RIPRENDI DA QUA, DA VERIFICARE SE FUNZIONA, INOLTRE RICONTROLLA SE EFFETIVAMENTE
# I TAG SONO FATTI BENE PERCHè QUESTO POTREBBE CAMBIARE TUTTO
# LINK UTILI: https://spacy.io/api/entityrecognizer#initialize
# https://stackoverflow.com/questions/66675261/how-can-i-work-with-example-for-nlp-update-problem-with-spacy3-0
#



# %%

def train_spacy(data, iterations,nlp):
    TRAINING_DATA = data
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner', last=True)


    # custom labels
    for _, ann in TRAINING_DATA:
        for ent in ann.get('entities'):
            ner.add_label(ent[2])

    # get other pipe components to disable during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itr in range(iterations):
            print("Starting Iteration: " + str(itr))
            random.shuffle(TRAINING_DATA)
            losses = {}
            example = []
            for text, ann in TRAINING_DATA:
                example.append(Example.from_dict(nlp.make_doc(text), ann))

            nlp.update(example, drop=0.2, sgd=optimizer, losses=losses)
            print(losses)

    return nlp



start_training = train_spacy(TRAINING_DATA, 100, nlp )
# %%
print(nlp.pipe_names)
nlp.to_disk('./custom_pipeline')
# %%
test = nlp("Patiente initialement admise à la clinique pour un épisode dépressif sévère. L'évolution s'est montrée peu favorable lors de cette hospitalisation. La patiente est donc transférée dans le service. La patiente s'est rapidement dégradée sur le plan somatique avec hyperthermie et signes neurologiques, notamment suite à un surdosage en augmentin ayant entraîné une insuffisance rénale aigue sur son insuffisance rénale chronique. Dans ce contexte, la patiente est transférée en service. Hospitalisée du 10/06/2021 au 23/06/2021 pour prise en charge de troubles de la vigilance fébriles. ")
colors = {"DRUG": "#F67DE3", "ADR": "#7DF6D9", "NLD":"#FFFFFF"}
options = {"colors": colors}

spacy.displacy.render(test, style="ent", options= options, jupyter=True)
print(nlp.pipe_names)