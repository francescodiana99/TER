#%%
import docx
from doc import *
import spacy
from spacy.tokens import DocBin
from sklearn.model_selection import KFold, train_test_split
from spacy.cli.train import train
from pathlib import Path
from metrics import get_scores
import pandas as pd
from cross_validation_alt import add_entity_ruler
from spacy.util import load_config
from metrics import plot_confusion_matrix, get_dataset_labels
from datetime import datetime

#%%

def display_results(nlp, test:list):
    colors = {"DRUG": "#F67DE3", "ADR": "#7DF6D9", "NLD":"#FFFFFF"}
    options = {"colors": colors}

    for ex in test:
        pred = nlp(ex.text)
        print("Prediction")
        spacy.displacy.render(pred, style="ent", options= options, jupyter=True)
        print("True labels")
        spacy.displacy.render(ex, style="ent", options= options, jupyter=True)

def DocBin_to_disk(data:list, path:str):
    """create a DocBin object from a list of annotations and saves it on disk"""

    db = DocBin()
    nlp = spacy.blank("fr")
    for text, annotations in data:
        doc = nlp(text)
        ents = []
        for start, end, label in annotations:

            span = doc.char_span(start, end, label=label)
            #print(span)
            ents.append(span)

        doc.ents = ents
        db.add(doc)
    db.to_disk(path)

def k_fold_split(data:list,n=5):
    """Applies k-fold split, saving .spacy files on disk"""
    kf = KFold(n_splits=n,shuffle=True, random_state=3)
    for i, (eval_index, train_index) in enumerate(kf.split(data)):
        eval_data = []
        training_data = []

        for j in train_index:
            training_data.append(data[j])

        for j in eval_index:
            eval_data.append(data[j])
        try:
            DocBin_to_disk(training_data, './cv/train_'+ str(i) + ".spacy")
            DocBin_to_disk(eval_data, './cv/eval_'+ str(i) + ".spacy")
            print("DocBin split " + str(i) + " created")

        except:
            print("Error in saving DocBin file")

#%%

document = docx.Document("./data/train1.docx")
document_2 = docx.Document("./data/train2.docx")


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
train_data, test_data = train_test_split(data, test_size=0.2)
DocBin_to_disk(test_data, "./cv/test.spacy")


n = 5
k_fold_split(data=train_data, n=n)
#%%
conf = Path("./configs/config_3.cfg")
output_path = Path("./models/model_16")
spacy.require_gpu(0)
precision = []
recall = []
f_score = []
for i in range(n):
    overrides = {"paths.train": "./cv/train_" + str(i) + ".spacy", "paths.dev": "./cv/eval_"+ str(i) + ".spacy"}
    train(conf, output_path, use_gpu=0, overrides=overrides)
    nlp = spacy.load("./models/model_16/model-last")
    db = DocBin().from_disk("./cv/eval_"+ str(i) + ".spacy")
    docs = list(db.get_docs(nlp.vocab))
    scores = get_scores(nlp, docs=docs)
    precision.append(scores["ents_p"])
    recall.append(scores["ents_r"])
    f_score.append(scores["ents_f"])

avg_r = sum(recall) / len(recall)
avg_p = sum(precision) / len(precision)
avg_f = sum(f_score) / len(f_score)

print("Precision: " + str(avg_p))
print("Recall: " + str(avg_r))
print("F-score: " + str(avg_f))

#%%
nlp = spacy.load("./models/model_16/model-best")
db = DocBin().from_disk("./cv/eval_0.spacy")
test_docs = list(db.get_docs(nlp.vocab))
test_scores = get_scores(nlp,test_docs)
config = load_config(conf)
results = {"Time": datetime.now(),
            "Name":"Model 16",
            "Ents":"DRUG,NLD,ADR",
            "Precision": avg_p,
            "Recall": avg_r,
            "F-score": avg_f,
            "Scores per label": str(test_scores["ents_per_type"]),
            "beta1": config["training"]["optimizer"]["beta1"],
            "beta2": config["training"]["optimizer"]["beta2"],
            "dropout": config["training"]["dropout"],
            "max_batch_items": config["components"]["transformer"]["max_batch_items"]

            }


#%%
plot_confusion_matrix(test_docs, classes=get_dataset_labels(test_docs), nlp=nlp, normalize=False)
df = pd.DataFrame([results])
try:
    df.to_csv('Results.csv', mode='a', index=False, header=False)
    print("Data appended successfully.")
except:
    print("Error in appending data")
df.style
#%%
# add Entity Ruler component
add_entity_ruler('./models/model_16/model-best', patterns='pattern.jsonl', path="./models/pipeline_16", before_ner=True)
nlp_ner = spacy.load("./models/pipeline_16")
print(nlp_ner.pipe_names)
#%%
# Test the pipeline with the entity ruler
nlp = spacy.load("./models/pipeline_16")

test_docs = list(db.get_docs(nlp.vocab))
test_scores = get_scores(nlp,test_docs)
config = load_config(conf)
results = {"Time" : datetime.now(),
            "Name":"Pipeline 16",
            "Ents":"ADR,NLD,DRUG",
            "Precision": test_scores["ents_p"],
            "Recall": str(test_scores["ents_r"]),
            "F-score": str(test_scores["ents_f"]),
            "Scores per label": str(test_scores["ents_per_type"]),
            "beta1": config["training"]["optimizer"]["beta1"],
            "beta2": config["training"]["optimizer"]["beta2"],
            "dropout": config["training"]["dropout"],
            "max_batch_items": config["components"]["transformer"]["max_batch_items"]
            }

plot_confusion_matrix(test_docs, classes=get_dataset_labels(test_docs), nlp=nlp, normalize=False)
df = pd.DataFrame([results])
try:
    df.to_csv('Results.csv', mode='a', index=False, header=False)
    print("Data appended successfully.")
except:
    print("Error in appending data")

#%%
# display results with displacy
display_results(nlp, test_docs)

