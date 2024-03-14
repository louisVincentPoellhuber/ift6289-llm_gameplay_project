import sys
import os
from dotenv import load_dotenv
load_dotenv()
CHATARENA_PATH = os.getenv("CHATARENA_PATH")
sys.path.append(CHATARENA_PATH)

from chatarena.agent import Player
from chatarena.backends import CohereAIChat
from chatarena.environments.askguess import AskGuess

import cohere

client = cohere.Client(api_key=os.environ.get("COHEREAI_API_KEY"))

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

env = AskGuess(player_names = ["Paya", "Toto"])
arena = Arena([paya, toto], env)
arena.launch_cli(interactive=False)