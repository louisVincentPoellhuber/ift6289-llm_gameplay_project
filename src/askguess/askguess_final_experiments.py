import os
import sys
import yaml
import cohere
from time import strftime
from dotenv import load_dotenv
import random
import json

# Configuring path from the library
CHATARENA_PATH = os.getenv("CHATARENA_PATH")
sys.path.append(CHATARENA_PATH)

# Loading dotenv
load_dotenv()

from chatarena.arena import Arena
from chatarena.agent import Player
from chatarena.backends import CohereAIChat
from chatarena.environments.askguess import AskGuess

# Prompts yaml
file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
chat_history_path = os.path.join(dir_path, "chat_history")
PROMPT_CONFIG_FILE = dir_path + "/askguess_final_experiments.yaml"

role_description = "You are a player in a word guessing game called ask-guess. "


with open(r"src\datasets\askguess.json", "r") as fp:
    word_list = json.load(fp)["wordict"]


with open(PROMPT_CONFIG_FILE) as file:
    prompts = yaml.safe_load(file)

prompt_mode = "final_baseline"

backend = CohereAIChat(
    temperature=prompts[prompt_mode]["temperature"],
    max_tokens=prompts[prompt_mode]["max_tokens"],
    model=prompts[prompt_mode]["model_name"],
    preamble=prompts[prompt_mode]["preamble"],
)

print("\n\n", "-----------------------------------")
print(f"PROMPT MODE:", prompt_mode, "\n\n")

# Defining players
paya = Player(name="Paya",
                role_desc=role_description,
                backend=CohereAIChat())

toto = Player(name="Toto",
                role_desc=role_description,
                backend=CohereAIChat())

for i in range(6):
    # Running experiment
    env = AskGuess(
        player_names = ["Paya", "Toto"], 
        word_list=word_list, 
        prompt_config_file=PROMPT_CONFIG_FILE,
        prompt_config_mode=prompt_mode,
        )

    arena = Arena([paya, toto], env)
    arena.launch_cli(interactive=False, max_steps=20)

    experiment_path = os.path.join(chat_history_path, prompt_mode)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)  # Create the directory if it doesn't exist

    # Saving history
    chat_path = os.path.join(
        experiment_path, f"askguess_{strftime('%Y_%m_%d_%H_%M_%S')}.json"
    )
    arena.save_chat(chat_path)
