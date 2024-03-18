import sys
import os
from dotenv import load_dotenv
load_dotenv()
CHATARENA_PATH = os.getenv("CHATARENA_PATH")
sys.path.append(CHATARENA_PATH)

import json
from time import strftime


from chatarena.agent import Player
from chatarena.backends import CohereAIChat, OpenAIChat, Claude
from chatarena.environments.askguess import AskGuess

import cohere
import openai
import anthropic

#client = cohere.Client(api_key=os.environ.get("COHEREAI_API_KEY"))
#client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

role_description = """
You are going to play in a two-player word guessing game. There are two roles in the game:
The speaker, who is given a secret word, and the guesser, who must guess the secret word. The 
speaker will give the guesser clues relating to the secret word. Together, the two must guess 
the word correctly in as few rounds as possible. 
"""

paya = Player(name="Paya",
                role_desc=role_description,
                backend=CohereAIChat())

toto = Player(name="Toto",
                role_desc=role_description,
                backend=CohereAIChat())

from chatarena.arena import Arena

with open(r"C:\Users\Louis\Documents\University\Masters\H24 - Deep NLP\ift6289-llm_gameplay_project\datasets\askguess.json", "r") as fp:
    word_list = json.load(fp)["wordict"]

env = AskGuess(player_names = ["Paya", "Toto"], word_list=word_list)
arena = Arena([paya, toto], env)
arena.launch_cli(interactive=False, max_steps=30)

# Saving history
arena.save_chat(f"src/askguess/chat_history/askguess_{strftime('%Y_%m_%d_%H_%M_%S')}.json")
