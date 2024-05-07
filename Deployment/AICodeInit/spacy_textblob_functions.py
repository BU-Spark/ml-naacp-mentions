from peft import AutoPeftModelForCausalLM
import spacy
import pandas as pd
from textblob import TextBlob


def load_model():
    nlp = spacy.load("en_core_web_sm") 
    return nlp

def extract_entities(text,nlp):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function to extract entities context
def extract_entities_with_context(text, nlp, window=5):
    doc = nlp(text)
    entity_context = []
    for ent in doc.ents:
        start = max(0, ent.start - window)
        end = min(len(doc), ent.end + window)
        context = doc[start:end].text
        entity_context.append((ent.text, ent.label_, context))
    return entity_context

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def analyze_entity_sentiments(entity_contexts):
    sentiments = []
    for text, label, context in entity_contexts:
        sentiment = get_sentiment(context)
        sentiments.append((text, label, sentiment))
    return sentiments

def analyze_entity_sentiments_score(entity_contexts):
    sentiments = []
    for text, label, context in entity_contexts:
        sentiment = get_sentiment(context)
        sentiments.append((sentiment))
    return sentiments

def calculate_avg_score(scores):
    if scores:
        return sum(scores) / len(scores)
    else:
        return float('inf')


def categorize_sentiment(score):
    if score <= -0.1:
        return 'Negative'
    elif score >= 0.1:
        return 'Positive'
    else:
        return 'Neutral'