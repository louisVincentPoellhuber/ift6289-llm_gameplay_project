import sys
import os
from dotenv import load_dotenv

load_dotenv()
CHATARENA_PATH = os.getenv("CHATARENA_PATH")
sys.path.append(CHATARENA_PATH)

import json
import yaml
from time import strftime
import random

from chatarena.arena import Arena
from chatarena.agent import Player
from chatarena.backends import CohereAIChat
from chatarena.environments.spyfall import SpyFall

role_description = """
You are going to play in a two-player word guessing game. There are two roles in the game:
The speaker, who is given a secret word, and the guesser, who must guess the secret word. The 
speaker will give the guesser clues relating to the secret word. Together, the two must guess 
the word correctly in as few rounds as possible. 
"""

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
chat_history_path = os.path.join(dir_path, "chat_history")
PROMPT_CONFIG_FILE = os.path.join(dir_path, "spyfall_prompt_config.yaml")

# Name of the players
number_of_players = 6

backend = CohereAIChat()

# Opening configurations
with open(PROMPT_CONFIG_FILE) as file:
    prompts = yaml.safe_load(file)


NB_EXPERIMENTS = 1
MAX_STEPS = 20
# PROMPT_MODES = ["baseline", "remember_json_guessing_format", "remember_sentence_format", "remember_bracket_format"]
PROMPT_MODES = ["bracket_chain_of_thoughts"]
# ================= EXPERIMENTS ================

for prompt_mode in PROMPT_MODES:
    print(f"======================= Doing {prompt_mode} =======================")

    for i in range(NB_EXPERIMENTS):

        # Defining players
        players = ["Nancy", "Tom", "Cindy", "Jack", "Rose", "Edward"][
            :number_of_players
        ]
        random.shuffle(players)
        # First description of the game -> taken from GameEval
        role_description = prompts["role_description"].format(
            number_of_players=len(players)
        )
        # Chatarena output format
        format_specification = prompts["json_format_specification"]
        if prompts[prompt_mode]["response_format"] == "string":
            format_specification = ""

        players_list = [
            Player(
                name=players[i],
                role_desc=role_description + format_specification,
                backend=backend,
            )
            for i in range(len(players[:number_of_players]))
        ]

        # Running experiment
        env = SpyFall(
            player_names=players,
            prompt_config_file=PROMPT_CONFIG_FILE,
            prompt_config_mode=prompt_mode,
        )
        arena = Arena(players_list, env)
        arena.launch_cli(interactive=False, max_steps=MAX_STEPS)

        experiment_path = os.path.join(chat_history_path, prompt_mode)
        if not os.path.exists(experiment_path):
            os.mkdir(experiment_path)  # Create the directory if it doesn't exist

        # Saving history
        chat_path = os.path.join(
            experiment_path, f"spyfall_{strftime('%Y_%m_%d_%H_%M_%S')}.json"
        )
        arena.save_chat(chat_path)
