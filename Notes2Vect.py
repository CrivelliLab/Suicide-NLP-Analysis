
# coding: utf-8

# # Notes 2 Vect
#
# The goal of this Notebook is to figure out if we can create an embedding for each specific note.

# In[54]:


#Preamble
import sys
import numpy as np
import gensim
import logging as log
import csv
import pickle as pkl
from nltk.tokenize import word_tokenize

# Logging config
log.basicConfig(format='%(levelname)s : %(message)s', level=log.INFO)

log.info("Running with vec_size={}".format(sys.argv[1]))

# Read in the .csv notes data:

# In[35]:


def load_notes():
    "Load in the notes from the .csv file"
    global notes

    file_name = 'data/disch_full.csv'

    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader) # skip the header
        notes = [row[3] for row in csv_reader]

    log.info("Done loading notes with length {}".format(len(notes)))


# In[56]:


def load_pickle_file(name):
    "Get the value from the pickle file and set it as the variable value"
    with open("data/{}.pkl".format(name), "rb") as pickle_file:
        globals()[name] = np.array(pkl.load(pickle_file))


# In[57]:


#load_pickle_file("tagged_notes")

# In[36]:


load_notes()


# ##  Vectors
#
# Now that all the notes are in the `notes` variable, we can begin training for our vectors. Reference material: [Article 1](https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5) and [article 2](https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.TaggedDocument).

# In[41]:


tagged_notes = [gensim.models.doc2vec.TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(notes)]

log.info("Loaded in tagged_notes")

# In[52]:

def save_pickle_file(name, data):
    "Save out the data as a pickle file"
    with open("data/{}.pkl".format(name), "wb") as pickle_file:
        pkl.dump(data, pickle_file)


# In[55]:


#save_pickle_file("tagged_notes", tagged_notes)


# In[50]:


def make_model(max_epochs = 5, vec_size = 100, alpha = 0.025):
    "Make the model for notes 2 vect"
    global model

    model = gensim.models.doc2vec.Doc2Vec(vector_size=vec_size,
                                            alpha=alpha,
                                            min_alpha=0.00025,
                                            min_count=1,
                                            dm=1,
                                            workers=2
                                         )
    model.build_vocab(tagged_notes)

    for epoch in range(max_epochs):
        log.info('iteration {0}'.format(epoch))
        model.train(tagged_notes,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("models/{}_{}_notes2vect.w2v".format(vec_size, max_epochs))
    log.info("Model Saved")


# In[51]:


make_model(vec_size = int(sys.argv[1]))

