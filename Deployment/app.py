from AICodeInit import *
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline
import gradio as gr
from huggingface_hub import login
import os

# Huggingface Authentication
api_key = os.getenv('cs549_naacp_access_key')

if api_key is None:
    raise ValueError("API_KEY is not set in the environment variables.")

login(api_key)

# Load the model
# model = AutoPeftModelForCausalLM.from_pretrained("Moritz-Pfeifer/financial-times-classification-llama-2-7b-v1.3")
# tokenizer = AutoTokenizer.from_pretrained("Moritz-Pfeifer/financial-times-classification-llama-2-7b-v1.3")

# def predict_text(test, model, tokenizer):
#     prompt = f"""
#             You are given a news article regarding the greater Boston area.
#             Analyze the sentiment of the article enclosed in square brackets,
#             determine if it is positive, negative or other, and return the answer as the corresponding sentiment label
#             "positive" or "negative". If the sentiment is neither positive or negative, return "other".

#             [{test}] ="""
#     pipe = pipeline(task="text-generation",
#                         model=model,
#                         tokenizer=tokenizer,
#                         max_new_tokens = 1,
#                         temperature = 0.1,
#                        )
#     result = pipe(prompt)
#     answer = result[0]['generated_text'].split("=")[-1]
#     # print(answer)
#     if "positive" in answer.lower():
#         return "positive"
#     elif "negative" in answer.lower():
#         return "negative"
#     else:
#         return "other"

# def predict(input_text):
#   return predict_text(input_text, model, tokenizer)


# interface = gr.Interface(fn=predict, inputs="text", outputs="text", title="Text Classifier", description="Insert your text and get the classification result.")
# interface.launch()

#initialize llama model model, tokenizer, prompt, pipe = create_pipeline()
nlp = load_model()
prompt, pipe = create_pipeline()
#demo function calls predict text from other file
def entity_sentiment(text):
    entities = set(extract_entities(text,nlp))
    entities_context = extract_entities_with_context(text, nlp)
    #sentiment = get_sentiment(text)
    entity_sentiment_scoreonly=analyze_entity_sentiments_score(entities_context)
    #sentiment_category = categorize_sentiment(sentiment)
    #average_sentiment = sum(entity_sentiments)/len(entity_sentiments)
    average_score = calculate_avg_score(entity_sentiment_scoreonly)
    average_sentiment = categorize_sentiment(average_score)

    llama_sentiment = predict_text(text,pipe,prompt)
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
    ])
    # output_label=["Entities","Entity Contexts", "Entity Sentiment Scores","Average Score of Entites","Llama sentiment on Lede"]

demo.launch(share=True)

if __name__ == "__main__":
    interface.launch(share=True)