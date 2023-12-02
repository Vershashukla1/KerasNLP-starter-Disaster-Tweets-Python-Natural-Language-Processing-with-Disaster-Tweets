# KerasNLP-starter -file-Disaster-Tweets-Python-Natural-Language-Processing-with-Disaster-Tweets
This starter file uses the DistilBERT pretrained model from KerasNLP.

BERT stands for Bidirectional Encoder Representations from Transformers. BERT and other Transformer encoder architectures have been wildly successful on a variety of tasks in NLP (natural language processing). They compute vector-space representations of natural language that are suitable for use in deep learning models.

The BERT family of models uses the Transformer encoder architecture to process each token of input text in the full context of all tokens before and after, hence the name: Bidirectional Encoder Representations from Transformers.

BERT models are usually pre-trained on a large corpus of text, then fine-tuned for specific tasks.

DistilBERT model is a distilled form of the BERT model. The size of a BERT model was reduced by 40% via knowledge distillation during the pre-training phase while retaining 97% of its language understanding abilities and being 60% faster.
In this File, you will-
 Load the Disaster Tweets
Explore the dataset
Preprocess the data
Load a DistilBERT model from Keras NLP
Train your own model, fine-tuning BERT
