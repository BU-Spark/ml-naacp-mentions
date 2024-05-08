import gradio as gr
from AICodeInit import *
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline


#initialize llama model model, tokenizer, prompt, pipe = create_pipeline()
nlp = load_model()
prompt, pipe = create_pipeline()
#demo function calls predict text from other file
def entity_sentiment(input):
    entities = extract_entities(input,nlp)
    entities_context = extract_entities_with_context(input, nlp)
    #sentiment = get_sentiment(text)
    entity_sentiment_scoreonly=analyze_entity_sentiments_score(entities_context)
    #sentiment_category = categorize_sentiment(sentiment)
    #average_sentiment = sum(entity_sentiments)/len(entity_sentiments)
    average_score = calculate_avg_score(entity_sentiment_scoreonly)
    average_sentiment = categorize_sentiment(average_score)

    llama_sentiment = predict_text(input,pipe,prompt)
    return entities,entities_context,entity_sentiment_scoreonly,average_score,average_sentiment,llama_sentiment

demo = gr.Interface(
    fn=entity_sentiment,
    inputs=["text"],
    outputs=[
        gr.Textbox(label="Entities"),
        gr.Textbox(label="Entity Contexts"),
        gr.Textbox(label="Entity Sentiment Scores"),
        gr.Textbox(label="Average Score"),
        gr.Textbox(label="Average Sentiment"),
        gr.Textbox(label="Llama Sentiment")
    ]
    # output_label=["Entities","Entity Contexts", "Entity Sentiment Scores","Average Score of Entites","Llama sentiment on Lede"]
)

demo.launch(share=True)
