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
from chatarena.environments.taboo import Taboo

import random

role_description = """
You are going to play in a two-player word guessing game. There are two roles in the game:
The speaker, who is given a secret word and some restricted words, and the guesser, who must guess the secret word. The 
speaker will give the guesser clues describing the secret word, but without mentioning the given restricted terms. Together, the two must guess 
the word correctly in as few rounds as possible. 
"""

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
PROMPT_CONFIG_FILE = os.path.join(dir_path, "taboo_prompt_config.yaml")

with open(PROMPT_CONFIG_FILE) as file:
    prompts = yaml.safe_load(file)


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


NB_EXPERIMENTS = 10
MAX_STEPS = 20
PROMPT_MODES = ["best"]
# ================= EXPERIMENTS ================

for prompt_mode in PROMPT_MODES:
    print(f"======================= Doing {prompt_mode} =======================")

    for i in range(NB_EXPERIMENTS):
        print(f"Experiment {i}.")

        # ====== Baseline =======
        env = Taboo(
            player_names = ["Paya", "Toto"], 
            taboo = taboo, 
            prompt_config_file=PROMPT_CONFIG_FILE,
            prompt_config_mode=prompt_mode,
            )

        arena = Arena([paya, toto], env)
        arena.launch_cli(interactive=False, max_steps=MAX_STEPS)

        # Saving history
        arena.save_chat(f"src/taboo/chat_history/{prompt_mode}/taboo_{strftime('%Y_%m_%d_%H_%M_%S')}.json")
