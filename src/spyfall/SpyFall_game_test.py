import os
import sys
import cohere
from datetime import datetime
from dotenv import load_dotenv
from chatarena.arena import Arena
from chatarena.agent import Player
from chatarena.backends import CohereAIChat
from chatarena.environments.spyfall import SpyFall

# Loading dotenv
load_dotenv()

# Configuring path from the library
CHATARENA_PATH = os.getenv("CHATARENA_PATH")
sys.path.append(CHATARENA_PATH)

# Starting cohere backend
client = cohere.Client(os.getenv("COHEREAI_API_KEY"))
backend = CohereAIChat()

# Name of the players
players = ["Nancy", "Tom", "Cindy", "Jack", "Rose", "Edward"]

# First description of the game -> taken from GameEval
role_description = """
Now you are playing the game called "who is the spy".

There are 6 players in this game. 
At the beginning of the game, everyone will receive a word.
There is one spy who will receive a spy word, while others will receive a common word.
Spy word is different but relevant to common words. For example, the spy word can be "apple", and the common word is "banana".

There are two stages in each round of the game.

The first stage is describing stage:
Everyone needs to use a word or a few words to say something in turn about the word he received without directly saying the word.
The funniest part of the game is that since you do not know other's words, you are not sure whether you are the spy.
So, you can only infer who have the different based on other players description.

The second stage is the voting stage:
After everyone speaks, each player must vote for a player who he thinks is the spy and tell others why. Players can not repeat other players' speeches or their own speeches in the previous round. The player getting the most votes will be eliminated from the game.

If the spy is eliminated, the rest of the players win.
If there are less than three players, and the spy still lives, then the spy wins.
"""

# Chatarena format
format_specification = """
Your output should be format in a json with the following schema:
```
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "word": {
      "description": "your secret word",
      "type": "string"
    },
    "arguments": {
      "description": "what you are going to say to your opponent",
      "type": "string"
    }
  },
  "required": ["word", "arguments"]
}
```

For example:
```
{\n  "word": "banana",\n  "arguments": "Is it a fruit that monkeys love." \n}

```
"""

# Creating players
player_0 = Player(
    name=players[0], role_desc=role_description + format_specification, backend=backend
)
player_1 = Player(
    name=players[1], role_desc=role_description + format_specification, backend=backend
)
player_2 = Player(
    name=players[2], role_desc=role_description + format_specification, backend=backend
)
player_3 = Player(
    name=players[3], role_desc=role_description + format_specification, backend=backend
)
player_4 = Player(
    name=players[4], role_desc=role_description + format_specification, backend=backend
)
player_5 = Player(
    name=players[5], role_desc=role_description + format_specification, backend=backend
)

# Player names
players = ["Nancy", "Tom", "Cindy", "Jack", "Rose", "Edward"]

# Running experiment
env = SpyFall(player_names=players)
arena = Arena([player_0, player_1, player_2, player_3, player_4, player_5], env)
arena.launch_cli(interactive=False)

# Saving history
arena.save_history("test1.json")
