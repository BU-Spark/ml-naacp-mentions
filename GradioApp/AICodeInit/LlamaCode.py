from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline
# Load the model

def create_pipeline():
    model = AutoPeftModelForCausalLM.from_pretrained("Moritz-Pfeifer/financial-times-classification-llama-2-7b-v1.3")
    tokenizer = AutoTokenizer.from_pretrained("Moritz-Pfeifer/financial-times-classification-llama-2-7b-v1.3")
    prompt = f"""
            "You are given a news article regarding the greater Boston area.
            Analyze the sentiment of the article enclosed in square brackets,
            determine if it is positive, negative or neutral and return the answer as the corresponding sentiment label
            "positive", "negative", or "neutral"".

            """
    pipe = pipeline(task="text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens = 1,
                        temperature = 0.1,
                        )
    return prompt, pipe

def predict_text(text,pipe,prompt):
    result = pipe((prompt+"\n"+'['+'{'+text+'}'+']'+' '+'='))
    answer = result[0]['generated_text'].split("=")[-1]
    if "positive" in answer.lower():
        return "positive"
    elif "negative" in answer.lower():
        return "negative"
    else:
        return "neutral"