import sys
import os
from dotenv import load_dotenv
load_dotenv()
CHATARENA_PATH = os.getenv("CHATARENA_PATH")
sys.path.append(CHATARENA_PATH)

import json
import yaml
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

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
PROMPT_CONFIG_FILE = os.path.join(dir_path, "askguess_prompt_config.yaml")

with open(PROMPT_CONFIG_FILE) as file:
    prompts = yaml.safe_load(file)


paya = Player(name="Paya",
                role_desc=role_description,
                backend=CohereAIChat())

toto = Player(name="Toto",
                role_desc=role_description,
                backend=CohereAIChat())

from chatarena.arena import Arena

with open(r"C:\Users\Louis\Documents\University\Masters\H24 - Deep NLP\ift6289-llm_gameplay_project\datasets\askguess.json", "r") as fp:
    word_list = json.load(fp)["wordict"]


NB_EXPERIMENTS = 10
MAX_STEPS = 20
PROMPT_MODES = ["best"]
# ================= EXPERIMENTS ================

for prompt_mode in PROMPT_MODES:
    print(f"======================= Doing {prompt_mode} =======================")

    for i in range(NB_EXPERIMENTS):
        print(f"Experiment {i}.")

        # ====== Baseline =======
        env = AskGuess(
            player_names = ["Paya", "Toto"], 
            word_list=word_list, 
            prompt_config_file=PROMPT_CONFIG_FILE,
            prompt_config_mode=prompt_mode,
            )

        arena = Arena([paya, toto], env)
        arena.launch_cli(interactive=False, max_steps=MAX_STEPS)

        # Saving history
        arena.save_chat(f"src/askguess/chat_history/{prompt_mode}/askguess_{strftime('%Y_%m_%d_%H_%M_%S')}.json")
