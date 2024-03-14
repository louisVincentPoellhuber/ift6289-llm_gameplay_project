import random
import re
from typing import Dict, List, Union

from ..agent import SIGNAL_END_OF_CONVERSATION
from ..message import Message, MessagePool
from .base import Environment, TimeStep, register_env
from ..utils import extract_jsons

DEFAULT_TOPIC_CODES = {
    "Fruits": [
        "Melon",
        "Watermelon",
    ],
    "Animals": [
        "Lion",
        "Tiger",
    ],
    "Sports": [
        "Soccer",
        "Basketball",
    ],
}

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


@register_env
class SpyFall(Environment):
    type_name = "spyfall"

    def __init__(
        self,
        player_names: List[str],
        topic_codes: Dict[str, List[str]] = None,
        **kwargs,
    ):
        super().__init__(player_names=player_names, topic_codes=topic_codes, **kwargs)

        if topic_codes is None:
            topic_codes = DEFAULT_TOPIC_CODES
        self.topic_codes = topic_codes

        # The "state" of the environment is maintained by the message pool
        self.message_pool = MessagePool()

        # Randomly sample a topic, code and spy player
        self.topic = None
        self.code = None
        self.spy_name = None
        self.non_spy_names = None

        # Game states
        self._current_turn = 0
        self._next_player_idx = 0
        self._current_phase = "give clues"  # "give clues", "accuse", "guess"
        self._players_votes = None
        self._initialized = False

        self.reset()  # To initialize the game (select topic, code, spy)

    def get_next_player(self) -> str:
        """Get the next player."""
        # if self._current_phase != "guess":
        #     return self.player_names[self._next_player_idx]
        # else:
        #     return self.spy_name
        return self.player_names[self._next_player_idx]

    def reset(self):
        """Sample topic, code and spy code."""
        self.topic = random.choice(list(self.topic_codes.keys()))
        self.code = random.choice(self.topic_codes[self.topic])
        self.spy_name = random.choice(self.player_names)
        self.non_spy_names = [
            name for name in self.player_names if name != self.spy_name
        ]
        self.spy_word = self.topic_codes[self.topic][0]
        self.non_spy_word = self.topic_codes[self.topic][1]

        self._current_turn = 0
        self._next_player_idx = 0
        self._current_phase = "give clues"

        self.message_pool.reset()

        self._moderator_speak(f"Host: The game now starts.")
        self._moderator_speak(
            f"Your word is: {self.non_spy_word}. Remember it for the next rounds and do not say it.",
            visible_to=self.non_spy_names,
        )
        self._moderator_speak(
            f"Your word is: {self.spy_word}. Remember it for the next rounds and do not say it.",
            visible_to=self.spy_name,
        )
        self._moderator_speak(
            "Host: Now it's the describing stage, players have to say something about the received word without directly saying it. "
            f"You cannot repeat what others has said. We will start with {self.player_names[0]}. "
            "Don't forget to answer in the correct JSON format. "
            "Do not say your word. "
        )
        self._current_turn = 1

        self._players_votes = {name: 0 for name in self.player_names}

        self._initialized = True
        init_timestep = TimeStep(
            observation=self.get_observation(),
            reward=self.get_zero_rewards(),
            terminal=False,
        )

        return init_timestep

    def print(self):
        self.message_pool.print()

    def get_observation(self, player_name=None) -> List[Message]:
        """Get observation for the player."""
        if player_name is None:
            return self.message_pool.get_all_messages()
        else:
            return self.message_pool.get_visible_messages(
                player_name, turn=self._current_turn
            )

    def _text2vote(self, text) -> str:
        """Convert text to vote, return a player's name."""
        # lower = text.lower().replace("[", "").replace("]", "").replace(".", "")
        text = text.lower()
        for name in self.player_names:
            candidates = [
                name.lower(),
                name.lower().replace(" ", ""),
                name.lower().replace(" ", "_"),
            ]
            if any([candidate in text for candidate in candidates]):
                return name
        return ""

    def _is_true_code(self, text) -> bool:
        """Check whether the text is the true code."""
        # Get the word enclosed by quote marks with regex
        pattern = r"\"(.+?)\""
        match = re.search(pattern, text)
        if match:
            return match.group(1).lower().replace(" ", "") == self.code.lower().replace(
                " ", ""
            )
        else:
            # if no quote marks, check whether the last k words match the code
            words = text.split()
            if len(words) >= len(self.code.split()):
                guessed_term = (
                    "".join(words[-len(self.code.split()) :]).lower().replace(".", "")
                )
                return guessed_term == self.code.lower().replace(" ", "").replace(
                    ".", ""
                )
            else:
                return False

    def _moderator_speak(self, text: str, visible_to: Union[str, List[str]] = "all"):
        """Moderator say something."""
        message = Message(
            agent_name="Moderator",
            content=text,
            turn=self._current_turn,
            visible_to=visible_to,
        )
        self.message_pool.append_message(message)

    def get_rewards(self, spy_win: bool) -> Dict[str, float]:
        """Get rewards for each player."""
        rewards = {}
        for name in self.player_names:
            # The winner gets 1, the loser gets 0
            rewards[name] = float((name == self.spy_name) == spy_win)

        return rewards

    def is_terminal(self) -> bool:
        """Check if the conversation is over."""
        # If the last message is the signal, then the conversation is over
        if self.message_pool.last_message.content.startswith(
            SIGNAL_END_OF_CONVERSATION
        ):
            return True

    def step(self, player_name: str, action: str) -> TimeStep:
        """
        Step function that is called by the arena.

        Args:
            player_name: the name of the player that takes the action
            action: the action that the agents wants to take
        """
        # If not initialized, reset the environment
        if not self._initialized:
            self.reset()

        # self.message_pool.print()
        # print(f"spy: {self.spy_name}, Code: {self.code}, Topic: {self.topic}")
        assert (
            player_name == self.get_next_player()
        ), f"Wrong player! It is {self.get_next_player()} turn."

        if self._current_phase == "give clues":

            ## PARSE RESPONSE see askguess
            # print("Content action:", action)
            json_list = extract_jsons(action)
            if len(json_list) != 1:
                raise ValueError(f"Player output {action} is not a valid json.")

            word = json_list[0].get("word", None)
            arguments = json_list[0].get("arguments", None)

            message = Message(
                agent_name=player_name, content=arguments, turn=self._current_turn
            )
            self.message_pool.append_message(message)

            # Update the counters
            self._current_turn += 1
            if self._next_player_idx < len(self.player_names) - 1:
                self._next_player_idx += 1
            else:
                self._next_player_idx = 0
                self._current_phase = "accuse"
                self._moderator_speak(
                    "Host: Now the voting start, please vote for the player you think is the spy and tell the reason why you think he is the spy."
                    "Do not say your word. " + format_specification
                )
                self._current_turn += 1

            timestep = TimeStep(
                observation=self.get_observation(),
                reward=self.get_zero_rewards(),
                terminal=False,
            )  # Return all the messages

        elif self._current_phase == "accuse":
            json_list = extract_jsons(action)
            if len(json_list) != 1:
                raise ValueError(f"Player output {action} is not a valid json.")

            word = json_list[0].get("word", None)
            arguments = json_list[0].get("arguments", None)

            message = Message(
                agent_name=player_name,
                content=arguments,
                turn=self._current_turn,
            )

            self.message_pool.append_message(message)
            vote = self._text2vote(arguments)

            if vote in self.player_names:
                self._players_votes[vote] += 1

            if self._next_player_idx < len(self.player_names) - 1:
                self._next_player_idx += 1
                rewards = self.get_zero_rewards()
                terminal = False
            else:
                # print(self._players_votes)
                rewards = self.get_zero_rewards()
                terminal = False

                accuse_correct, even_vote = True, False
                max_vote_player = max(self._players_votes, key=self._players_votes.get)
                # detach if other players has the same number of votes
                for name, vote in self._players_votes.items():
                    if (
                        name != max_vote_player
                        and vote == self._players_votes[max_vote_player]
                    ):
                        accuse_correct, even_vote = False, True
                if max_vote_player != self.spy_name:
                    accuse_correct = False

                if not accuse_correct:
                    if even_vote:
                        self._moderator_speak(
                            f"There are even votes. The accusation does not stand."
                            # f"{self.spy_name} is the spy. {self.spy_name} won the game!"
                        )

                    else:
                        self._moderator_speak(
                            f"The most-voted player is {max_vote_player}. The accusation is incorrect."
                            # f"{self.spy_name} is the spy. {self.spy_name} won the game!"
                        )

                        # remove the max_vote_player
                        print(f"Removing player {max_vote_player}")
                        self.player_names.remove(max_vote_player)

                    # change the game self._current_phase == "give clues"
                    self._current_phase == "give clues"
                    self._next_player_idx = 0

                    self._moderator_speak(
                        "Host: Now it's the describing stage, players have to say something about the received word without directly saying it. "
                        f"You cannot repeat what others has said. "
                        "Do not say your word. " + format_specification
                    )

                    if len(self.player_names) < 4:
                        rewards = self.get_rewards(spy_win=True)
                        terminal = True
                else:
                    self._moderator_speak(
                        f"The accusation is correct! {self.spy_name} is the spy! "
                        f"Now {self.spy_name} can guess the secret code. "
                        'You should say: I guess the code is "..."'
                    )

                self._current_turn += 1

            timestep = TimeStep(
                observation=self.get_observation(), reward=rewards, terminal=terminal
            )

        # Check if the player signals the end of the conversation
        if self.is_terminal():
            timestep.terminal = True

        return timestep
