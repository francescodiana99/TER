
#%%

from sklearn.metrics import confusion_matrix
import spacy
from spacy.tokens import DocBin
from spacy.training import offsets_to_biluo_tags
from matplotlib import pyplot
import numpy as np
from spacy.training import Example
from spacy.scorer import Scorer
import json

def get_cleaned_label(label:str):
    """Removes the "-" symbol from the BILOU tag, extracting only the token name"""
    if "-" in label:
        return label.split("-")[1]
    else:
        return label

def create_total_target_vector(docs: list):
    """Given a list of Doc objects, returns a list of token enyity names"""
    target_vector = []
    for doc in docs:
        #print(str((doc.ents[0].start_char, doc.ents[0].end_char, doc.ents[0].label_)))
        bilou_enities = offsets_to_biluo_tags(doc,[(e.start_char, e.end_char, e.label_) for e in doc.ents])
        final = []
        for item in bilou_enities:
            final.append(get_cleaned_label(item))
        target_vector.extend(final)
    return target_vector

def create_prediction_vector(text, nlp):
    """Returns a list of token predictions, given in input a text"""
    return [get_cleaned_label(prediction) for prediction in get_all_ner_predictions(text, nlp=nlp)]

def create_total_prediction_vector(docs:list, nlp):
    """Returns a list of all tokens prediction, given in input a list of docs"""
    prediction_vector = []
    for doc in docs:
        prediction_vector.extend(create_prediction_vector(doc.text, nlp=nlp))
    return prediction_vector

def get_all_ner_predictions(text, nlp):
    "Returns a list of predictions in the BILOU form"
    doc = nlp(text)
    entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    bilou_entities = offsets_to_biluo_tags(doc, entities)
    return bilou_entities

def get_model_labels():
    "Returns a list of labels"
    labels = list(nlp.get_pipe("ner").labels)
    # add a label for tokens that are not part of any entity
    labels.append("O")
    return sorted(labels)

def get_dataset_labels(docs):
    return sorted(set(create_total_target_vector(docs)))

def generate_confusion_matrix(docs, nlp):
    """Returns a confusion matrics"""
    classes = sorted(set(create_total_target_vector(docs)))
    y_true = create_total_target_vector(docs)
    y_pred = create_total_prediction_vector(docs, nlp=nlp)
    # print (y_true)
    # print (y_pred)
    return confusion_matrix(y_true, y_pred, labels=classes)

def plot_confusion_matrix(docs, classes, nlp, normalize=False, cmap=pyplot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    title = "Confusion Matrix"

    # Compute confusion matrix
    cm = generate_confusion_matrix(docs, nlp=nlp)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = pyplot.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks= np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
   # Rotate the tick labels and set their alignment.
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and cretìate text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return cm, ax, pyplot

def get_scores(nlp, docs: list):
    """Returns a dictionary containing scores given in input a list of evaluation Doc objects """
    examples = []
    for doc in docs:
        prediction = nlp(doc.text)
        reference = doc
        example = Example(prediction, reference)
        examples.append(example)
        scorer = Scorer()
        scores = scorer.score(examples=examples)

    return scores

def results_to_json(results: list, path:str):
    """Saves the predictions in .json format"""
    docs = {}
    i = 0
    for result in results:
        anns = {}
        anns["text"] = result.text
        j = 0
        doc = {}
        for ents in result.ents:
            doc["id"] = j
            doc["text"] = ents.text
            doc["start"] = ents.start_char
            doc["end"] = ents.end_char
            doc["label"] = ents.label_
            j = j + 1
        anns["ents"] = doc
        docs[str(i)] = anns
        i = i + 1

    try:
        f = open(path, "w")
        json.dump(docs, f, indent=6, ensure_ascii=False)
        f.close()
        print("File saved: "+ path)
    except:
        print("Error in saving the JSON file")






#%%
if __name__ == "__main__":
   from sklearn.metrics import confusion_matrix
   import spacy
   from spacy.tokens import DocBin
   from spacy.training import offsets_to_biluo_tags


   nlp = spacy.load('./models/model_15/model-best')
   db = DocBin().from_disk("./cv_alt/test.spacy")
   docs = list(db.get_docs(nlp.vocab))

   plot_confusion_matrix(docs, classes= get_dataset_labels(),nlp=nlp,  normalize=False )

   scores = get_scores(nlp, docs=docs)
   print("Precision: " +  str(scores["ents_p"]))
   print("Recall: " +  str(scores["ents_r"]))
   print("F-score: " + str(scores["ents_f"]))

   print("Scores per type: " + str(scores["ents_per_type"]))
   results_to_json(docs,"./predictions.json")





# %%
