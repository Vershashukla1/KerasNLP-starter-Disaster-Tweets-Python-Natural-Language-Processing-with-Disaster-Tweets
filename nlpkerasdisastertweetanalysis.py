# %% [markdown]
# <center><img src="https://keras.io/img/logo-small.png" alt="Keras logo" width="100"><br/>
# This starter notebook is provided by the Keras team.</center>
# 
# ## Keras NLP starter guide here: https://keras.io/guides/keras_nlp/getting_started/
# 
# In this competition, the challenge is to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t.
# A dataset of 10,000 tweets that were hand classified is available. 
# 
# __This starter notebook uses the [DistilBERT](https://arxiv.org/abs/1910.01108) pretrained model from KerasNLP.__
# 
# 
# **BERT** stands for **Bidirectional Encoder Representations from Transformers**. BERT and other Transformer encoder architectures have been wildly successful on a variety of tasks in NLP (natural language processing). They compute vector-space representations of natural language that are suitable for use in deep learning models.
# 
# The BERT family of models uses the **Transformer encoder architecture** to process each token of input text in the full context of all tokens before and after, hence the name: Bidirectional Encoder Representations from Transformers.
# 
# BERT models are usually pre-trained on a large corpus of text, then fine-tuned for specific tasks.
# 
# **DistilBERT model** is a distilled form of the **BERT** model. The size of a BERT model was reduced by 40% via knowledge distillation during the pre-training phase while retaining 97% of its language understanding abilities and being 60% faster.
# 
# 
# 
# ![BERT Architecture](https://www.cse.chalmers.se/~richajo/nlp2019/l5/bert_class.png)
# 
# 
# 
# In this notebook, you will:
# 
# - Load the Disaster Tweets
# - Explore the dataset
# - Preprocess the data
# - Load a DistilBERT model from Keras NLP
# - Train your own model, fine-tuning BERT
# - Generate the submission file
# 

# %% [code] {"execution":{"iopub.status.busy":"2023-12-02T04:29:42.288499Z","iopub.execute_input":"2023-12-02T04:29:42.288855Z","iopub.status.idle":"2023-12-02T04:30:07.939827Z","shell.execute_reply.started":"2023-12-02T04:29:42.288821Z","shell.execute_reply":"2023-12-02T04:30:07.938642Z"}}
!pip install keras-core --upgrade
!pip install -q keras-nlp --upgrade

# This sample uses Keras Core, the multi-backend version of Keras.
# The selected backend is TensorFlow (other supported backends are 'jax' and 'torch')
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

# %% [code] {"execution":{"iopub.status.busy":"2023-12-02T04:30:07.941895Z","iopub.execute_input":"2023-12-02T04:30:07.942215Z","iopub.status.idle":"2023-12-02T04:30:17.283949Z","shell.execute_reply.started":"2023-12-02T04:30:07.942185Z","shell.execute_reply":"2023-12-02T04:30:17.283024Z"}}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras_core as keras
import keras_nlp
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("KerasNLP version:", keras_nlp.__version__)

# %% [markdown]
# # Load the Disaster Tweets
# Let's have a look at the train and test dataset.
# 
# They contain:
# - id
# - keyword: A keyword from that tweet (although this may be blank!)
# - location: The location the tweet was sent from (may also be blank)
# - text: The text of a tweet
# - target: 1 if the tweet is a real disaster or 0 if not

# %% [code] {"execution":{"iopub.status.busy":"2023-12-02T04:30:17.670564Z","iopub.execute_input":"2023-12-02T04:30:17.670838Z","iopub.status.idle":"2023-12-02T04:30:17.707256Z","shell.execute_reply.started":"2023-12-02T04:30:17.670814Z","shell.execute_reply":"2023-12-02T04:30:17.706385Z"}}
df_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

print('Training Set Shape = {}'.format(df_train.shape))
print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))
print('Test Set Shape = {}'.format(df_test.shape))
print('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))

# %% [code] {"execution":{"iopub.status.busy":"2023-12-02T04:30:17.883173Z","iopub.execute_input":"2023-12-02T04:30:17.883800Z","iopub.status.idle":"2023-12-02T04:30:17.898178Z","shell.execute_reply.started":"2023-12-02T04:30:17.883766Z","shell.execute_reply":"2023-12-02T04:30:17.897233Z"}}
df_train.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-12-02T04:30:35.169490Z","iopub.execute_input":"2023-12-02T04:30:35.169856Z","iopub.status.idle":"2023-12-02T04:30:35.180767Z","shell.execute_reply.started":"2023-12-02T04:30:35.169824Z","shell.execute_reply":"2023-12-02T04:30:35.179790Z"}}
df_test.head()

# %% [markdown]
# # Explore the dataset

# %% [code] {"execution":{"iopub.status.busy":"2023-12-02T04:30:52.292156Z","iopub.execute_input":"2023-12-02T04:30:52.292552Z","iopub.status.idle":"2023-12-02T04:30:52.318119Z","shell.execute_reply.started":"2023-12-02T04:30:52.292518Z","shell.execute_reply":"2023-12-02T04:30:52.316909Z"}}
df_train["length"] = df_train["text"].apply(lambda x : len(x))
df_test["length"] = df_test["text"].apply(lambda x : len(x))

print("Train Length Stat")
print(df_train["length"].describe())
print()

print("Test Length Stat")
print(df_test["length"].describe())

# %% [markdown]
# If you want to know more information about the data, you can grab useful information [here](https://www.kaggle.com/code/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert)
# 
# Note that all the tweets are in english.

# %% [markdown]
# # Preprocess the data

# %% [code] {"execution":{"iopub.status.busy":"2023-12-02T04:33:05.970499Z","iopub.execute_input":"2023-12-02T04:33:05.970840Z","iopub.status.idle":"2023-12-02T04:33:05.975992Z","shell.execute_reply.started":"2023-12-02T04:33:05.970813Z","shell.execute_reply":"2023-12-02T04:33:05.974957Z"}}
BATCH_SIZE = 64
NUM_TRAINING_EXAMPLES = df_train.shape[0]
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.3
STEPS_PER_EPOCH = int(NUM_TRAINING_EXAMPLES)*TRAIN_SPLIT // BATCH_SIZE

EPOCHS = 50
AUTO = tf.data.experimental.AUTOTUNE

# %% [code] {"execution":{"iopub.status.busy":"2023-12-02T04:33:39.881202Z","iopub.execute_input":"2023-12-02T04:33:39.881884Z","iopub.status.idle":"2023-12-02T04:33:39.889987Z","shell.execute_reply.started":"2023-12-02T04:33:39.881848Z","shell.execute_reply":"2023-12-02T04:33:39.889069Z"}}
from sklearn.model_selection import train_test_split

X = df_train["text"]
y = df_train["target"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SPLIT, random_state=100)

X_test = df_test["text"]

# %% [markdown]
# # Load a DistilBERT model from Keras NLP
# 
# Text inputs need to be transformed to numeric token ids and arranged in several Tensors before being input to BERT.
# 
# The BertClassifier model can be configured with a preprocessor layer, in which case it will automatically apply preprocessing to raw inputs during fit(), predict(), and evaluate(). This is done by default when creating the model with from_preset().
# 
# We will choose DistilBERT model.that learns a distilled (approximate) version of BERT, retaining 97% performance but using only half the number of parameters ([paper](https://arxiv.org/abs/1910.01108)). 
# 
# It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT’s performances as measured on the GLUE language understanding benchmark.
# 
# Specifically, it doesn't have token-type embeddings, pooler and retains only half of the layers from Google's BERT.

# %% [code] {"execution":{"iopub.status.busy":"2023-12-02T04:33:56.817411Z","iopub.execute_input":"2023-12-02T04:33:56.817777Z","iopub.status.idle":"2023-12-02T04:34:13.891810Z","shell.execute_reply.started":"2023-12-02T04:33:56.817745Z","shell.execute_reply":"2023-12-02T04:34:13.890954Z"}}
# Load a DistilBERT model.
preset= "distil_bert_base_en_uncased"

# Use a shorter sequence length.
preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(preset,
                                                                   sequence_length=160,
                                                                   name="preprocessor_4_tweets"
                                                                  )

# Pretrained classifier.
classifier = keras_nlp.models.DistilBertClassifier.from_preset(preset,
                                                               preprocessor = preprocessor, 
                                                               num_classes=2)

classifier.summary()

# %% [markdown]
# # Train your own model, fine-tuning BERT

# %% [code] {"execution":{"iopub.status.busy":"2023-12-02T04:46:29.687643Z","iopub.execute_input":"2023-12-02T04:46:29.688450Z","iopub.status.idle":"2023-12-02T05:22:44.750989Z","shell.execute_reply.started":"2023-12-02T04:46:29.688410Z","shell.execute_reply":"2023-12-02T05:22:44.749791Z"}}
# Compile
classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), #'binary_crossentropy',
    optimizer=keras.optimizers.Adam(1e-3),
    metrics= ["accuracy"]  
)

# Fit
history = classifier.fit(x=X_train,
                         y=y_train,
                         batch_size=BATCH_SIZE,
                         epochs=EPOCHS, 
                         validation_data=(X_val, y_val)
                        )

# %% [code] {"execution":{"iopub.status.busy":"2023-12-02T05:30:50.158911Z","iopub.execute_input":"2023-12-02T05:30:50.159312Z","iopub.status.idle":"2023-12-02T05:30:50.166064Z","shell.execute_reply.started":"2023-12-02T05:30:50.159281Z","shell.execute_reply":"2023-12-02T05:30:50.164996Z"}}
def displayConfusionMatrix(y_true, y_pred, dataset):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        np.argmax(y_pred, axis=1),
        display_labels=["Not Disaster","Disaster"],
        cmap=plt.cm.Blues
    )

    tn, fp, fn, tp = confusion_matrix(y_true, np.argmax(y_pred, axis=1)).ravel()
    f1_score = tp / (tp+((fn+fp)/2))

    disp.ax_.set_title("Confusion Matrix on " + dataset + " Dataset -- F1 Score: " + str(f1_score.round(2)))


# %% [code] {"execution":{"iopub.status.busy":"2023-12-02T05:30:54.659041Z","iopub.execute_input":"2023-12-02T05:30:54.659405Z","iopub.status.idle":"2023-12-02T05:31:15.966805Z","shell.execute_reply.started":"2023-12-02T05:30:54.659374Z","shell.execute_reply":"2023-12-02T05:31:15.965899Z"}}
y_pred_train = classifier.predict(X_train)

displayConfusionMatrix(y_train, y_pred_train, "Training")

# %% [code] {"execution":{"iopub.status.busy":"2023-12-02T05:31:56.257184Z","iopub.execute_input":"2023-12-02T05:31:56.257542Z","iopub.status.idle":"2023-12-02T05:32:06.386268Z","shell.execute_reply.started":"2023-12-02T05:31:56.257514Z","shell.execute_reply":"2023-12-02T05:32:06.385304Z"}}
y_pred_val = classifier.predict(X_val)

displayConfusionMatrix(y_val, y_pred_val, "Validation")

# %% [markdown]
# # Generate the submission file 
# 
# For each tweets in the test set, we predict if the given tweet is about a real disaster or not. If so, predict a 1. If not, predict a 0.
# 
# The `submission.csv` file uses the following format:
# `id,target`

# %% [code] {"execution":{"iopub.status.busy":"2023-12-02T05:32:22.057756Z","iopub.execute_input":"2023-12-02T05:32:22.058636Z","iopub.status.idle":"2023-12-02T05:32:22.075347Z","shell.execute_reply.started":"2023-12-02T05:32:22.058599Z","shell.execute_reply":"2023-12-02T05:32:22.074474Z"}}
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-12-02T05:32:30.983633Z","iopub.execute_input":"2023-12-02T05:32:30.984679Z","iopub.status.idle":"2023-12-02T05:32:43.471187Z","shell.execute_reply.started":"2023-12-02T05:32:30.984628Z","shell.execute_reply":"2023-12-02T05:32:43.470346Z"}}
sample_submission["target"] = np.argmax(classifier.predict(X_test), axis=1)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-02T05:33:42.073489Z","iopub.execute_input":"2023-12-02T05:33:42.073871Z","iopub.status.idle":"2023-12-02T05:33:42.092747Z","shell.execute_reply.started":"2023-12-02T05:33:42.073840Z","shell.execute_reply":"2023-12-02T05:33:42.091906Z"}}
sample_submission.describe()

# %% [code] {"execution":{"iopub.status.busy":"2023-12-02T05:34:05.090946Z","iopub.execute_input":"2023-12-02T05:34:05.091510Z","iopub.status.idle":"2023-12-02T05:34:05.103144Z","shell.execute_reply.started":"2023-12-02T05:34:05.091478Z","shell.execute_reply":"2023-12-02T05:34:05.102291Z"}}
sample_submission.to_csv("submission.csv", index=False)