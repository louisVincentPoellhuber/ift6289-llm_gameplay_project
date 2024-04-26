import sys
import os
from dotenv import load_dotenv
load_dotenv()
CHATARENA_PATH = os.getenv("CHATARENA_PATH")
sys.path.append(CHATARENA_PATH)

import json
from time import strftime


from chatarena.agent import Player
from chatarena.backends import CohereAIChat
from chatarena.environments.taboo import Taboo

import cohere
import random

client = cohere.Client(api_key=os.environ.get("COHEREAI_API_KEY"))

role_description = """
You are going to play in a two-player word guessing game. There are two roles in the game:
The speaker, who is given a secret word and some restricted words, and the guesser, who must guess the secret word. The 
speaker will give the guesser clues describing the secret word, but without mentioning the given restricted terms. Together, the two must guess 
the word correctly in as few rounds as possible. 
"""

paya = Player(name="Paya",
                role_desc=role_description,
                backend=CohereAIChat())

toto = Player(name="Toto",
                role_desc=role_description,
                backend=CohereAIChat())

from chatarena.arena import Arena

datasets = os.listdir(r"src\datasets\taboo")
random_dataset = random.choice(datasets)
with open(rf"src\datasets\taboo\{random_dataset}", "r") as fp:
    taboo = json.load(fp)

env = Taboo(player_names = ["Paya", "Toto"], taboo = taboo)
arena = Arena([paya, toto], env)
arena.launch_cli(interactive=False, max_steps=10)

# Saving history
arena.save_history(
    f"src/taboo/chat_history/taboo_{strftime('%Y_%m_%d_%H_%M_%S')}.json"
)