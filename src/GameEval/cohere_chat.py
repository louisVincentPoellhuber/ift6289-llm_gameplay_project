import sys
import openai
from func_timeout import func_set_timeout

from chat.config import key_gpt3, api_type_gpt3, api_base_gpt3, api_version_gpt3, engine_gpt3, temperature_gpt3

@func_set_timeout(15)
def get_response(messages):
    response = co.chat(
    chat_history=[
        {"role": "USER", "message": "Who discovered gravity?"},
        {"role": "CHATBOT", "message": "The man who is widely credited with \
        discovering gravity is Sir Isaac Newton"}
    ],
    message="What year was he born?",
    # perform web search before answering the question. You can also use your own custom connector.
    connectors=[{"id": "web-search"}]
    )
    return response

class COHERE:
    def __init__(self) -> None:
        self.name = "cohere"
        self.co = cohere.Client(os.getenv("COHERE_API_KEY"))

    def single_chat(self,content,role=None):
        if role is None:
            role = "You are an AI assistant that helps people find information."
        chat_history = [
                    {"role":"USER","message":role},
                    {"role":"USER","message":content}
                    ]
        res = None
        cnt = 0
        while True:
            try:
                response = get_response(messages)
                res = response["choices"][0]["message"]["content"]
                break
            except:
                cnt += 1 
            if cnt >= 5:
                break    
       
        return  res

    def multi_chat(self, messages):

        res = None
        cnt = 0

        while True:
            try:
                response = get_response(messages)
                res = response["choices"][0]["message"]["content"]
                break
            except:
                cnt += 1 
            if cnt >= 3:
                break    

        return  res
    
if __name__ == "__main__":
    pass