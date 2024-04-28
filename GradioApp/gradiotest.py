import gradio as gr
from AICodeInit import *
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline


#initialize llama model model, tokenizer, prompt, pipe = create_pipeline()
nlp = load_model()
prompt, pipe = create_pipeline()
#demo function calls predict text from other file
def entity_sentiment(text):
    entities = extract_entities(text,nlp)
    entity_context_list = extract_entities_with_context(text, nlp)
    sentiment = get_sentiment(text)
    entity_sentiments=analyze_entity_sentiments(entity_context_list)
    sentiment_category = categorize_sentiment(sentiment)
    llama_sentiment = predict_text(text,pipe,prompt)
    return entities,entity_context_list,sentiment,entity_sentiments,sentiment_category, llama_sentiment

demo = gr.Interface(
    fn=entity_sentiment,
    inputs=["text"],
    outputs=["text","text","text","text","text","text"],
)

demo.launch(share=True)
