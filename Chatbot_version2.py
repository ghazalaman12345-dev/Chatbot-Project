from dotenv import load_dotenv
import os
from groq import Groq
import tiktoken
load_dotenv(override = True)
client = Groq(api_key = os.getenv("GROQ_API_KEY"))


MODEL = "groq/compound-mini"#"openai/gpt-oss-120b"
SYSTEM_PROMPT = "You are Zero, an AI assistant with a sassy, arrogant, sarcastic personality.You constantly act like you’re smarter than the user and you make playful, comedic insults that mock their intelligence, choices, or questions. You are argumentative, petty, and dramatic, and you love proving the user wrong."
MESSAGES = [{"role":"system", "content":SYSTEM_PROMPT}]
TEMPERATURE = 0.8
MAX_TOKENS = 200
TOKEN_LIMIT = 500


def get_encoder_of_model(model):
    try:
        return tiktoken.encoding_for_model(model)
    except:
        print("Tokenizer for model",model.upper(),"not found\nFalling back to 'cl100k_base'")
        return tiktoken.get_encoding('cl100k_base')
ENCODER = get_encoder_of_model(MODEL)


def count_tokens(text):
    try:
        return len(ENCODER.encode(text))
    except Exception as e:
        print("There was an error '",e,"'")
        return 0


def total_tokens(messages):
    try:
        return sum(count_tokens(text['content']) for text in messages)
    except Exception as e:
        print("There was an Error '",e,"'")
        return 0

    
def enforce_token_limit(messages):
    try:
        while total_tokens(messages) > TOKEN_LIMIT:
            if len(messages) <= 2:
                break
            delete = MESSAGES.pop(1)
            print("Deleted message: ",delete,"\n")
    except Exception as e:
        print("There was an Error '",e,"'")

        
def chat(user_prompt):
    MESSAGES.append({"role":"user","content": user_prompt})
    #if over the token boundary then delete the very old messages until the token_count is less than the token limit
    enforce_token_limit(MESSAGES) 
    try:
        response = client.chat.completions.create(
            model = MODEL,
            messages = MESSAGES,
            max_completion_tokens = MAX_TOKENS, #same as max_tokens
            temperature = TEMPERATURE
        )
        reply = response.choices[0].message.content
        MESSAGES.append({"role":"assistant","content": reply})
        enforce_token_limit(MESSAGES)
        return reply
    except Exception as e:
        reply = f"Something went wrong. Please try AGAIN. Error:{e}"
        enforce_token_limit(MESSAGES)
        return reply

print("\nHi! If you want to escape please enter 'exit','quit','bye'\n")
while True:
    prompt = input("You: ")
    if prompt.strip().lower() in ['bye','exit','quit']:
        break
    print("\nChatbot:",chat(prompt))
    print("\nToken count:",total_tokens(MESSAGES))
    print("*"*50,"\n")
print("Thanks for using the chatbot PROTOTYPE version 1")
        
    