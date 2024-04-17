import yaml
import random
import re
from typing import Dict, List, Union

from ..agent import SIGNAL_END_OF_CONVERSATION
from ..message import Message, MessagePool
from .base import Environment, TimeStep, register_env
from ..utils import extract_jsons_spyfall


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
    "Transport": [
        "Airplane",
        "Hellicopter",
    ],
    "Official places": [
        "Police Station",
        "Fire",
    ],
    "Instruments": [
        "Electic Guitar",
        "Acoustic Guitar",
    ],
    "Furniture": [
        "Sofa",
        "Chair",
    ],
    "Tools": [
        "Hammer",
        "Screwdriver",
    ],
}


@register_env
class SpyFall(Environment):
    type_name = "spyfall"

    def __init__(
        self,
        prompt_config_mode: str,
        prompt_config_file: str,
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
        self._end_condition = None
        self._prompt_config_mode = prompt_config_mode
        self._prompt_config_prompt_config_file = prompt_config_file

        # number of roungs
        self._ONE_ROUND = False

        # Reading prompt configs
        with open(self._prompt_config_prompt_config_file, "r") as file:
            self._prompts = yaml.safe_load(file)

        self.reset()  # To initialize the game (select topic, code, spy)

    def get_next_player(self) -> str:
        """Get the next player."""
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

        self._moderator_speak(f"The game now starts.")
        # Moderator gives word to villagers
        self._moderator_speak(
            self._prompts[self._prompt_config_mode]["moderator_gives_word"]
            .format(word=self.non_spy_word)
            .replace(r"{{", "{")
            .replace(r"}}", "}"),
            visible_to=self.non_spy_names,
        )
        # Moderator gives word to spy
        self._moderator_speak(
            self._prompts[self._prompt_config_mode]["moderator_gives_word"]
            .format(word=self.spy_word)
            .replace(r"{{", "{")
            .replace(r"}}", "}"),
            visible_to=self.spy_name,
        )
        # Moderator says it's describing phase
        self._moderator_speak(
            self._prompts[self._prompt_config_mode]["describing_phase"]
            .format(player=self.player_names[0])
            .replace(r"{{", "{")
            .replace(r"}}", "}")
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
        # text = text.lower()
        # print(text)
        try:
            found_names = [candidate in text for candidate in self.player_names]
            if any(found_names):
                for i, status in enumerate(found_names):
                    if status:
                        return self.player_names[i]
        except:
            print("No name found")
            print("Text:", text)
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
        try:
            if self.message_pool.last_message.content.startswith(
                SIGNAL_END_OF_CONVERSATION
            ):
                print("============ SIGNAL_END_OF_CONVERSATION ============")
                return True
        except AttributeError:
            return True

    def get_disposition(self) -> Dict:
        disposition = {}
        disposition["true_word"] = self.non_spy_word
        disposition["spy_word"] = self.spy_word
        disposition["nb_players"] = len(self.player_names)
        disposition["roles"] = {
            "non_spy": self.non_spy_names,
            "spy": [self.spy_name],
        }  # has to be lists!
        disposition["prompt_version"] = self._prompt_config_mode
        return disposition

    def get_metrics(self) -> Dict:
        metrics = {}

        metrics["nb_turns"] = self._current_turn

        if self._end_condition != None:
            metrics["end_condition"] = (
                self._end_condition
            )  # Ending error, Chat error or Answer Mentioned Error
        else:
            metrics["end_condition"] = (
                "RLE"  # Round Limit Error (reaches the number of max steps)
            )

        return metrics

    def _get_word_and_argument(self, action, json_list, response_format):
        if response_format == "json":
            try:
                word = json_list[0]["properties"]["word"]["description"]
                arguments = json_list[0]["properties"]["arguments"]["description"]
            except:
                print("JSON ERROR")
        elif response_format == "string":
            word = None
            arguments = action
            # print("##################")
            # print(action)
            # print("##################")
        return word, arguments

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
        # print(f"spy: {self.spy_name}")
        assert (
            player_name == self.get_next_player()
        ), f"Wrong player! It is {self.get_next_player()} turn."

        if self._current_phase == "give clues":

            ## PARSE RESPONSE see askguess
            json_list = []
            if self._prompts[self._prompt_config_mode]["response_format"] == "json":
                json_list = extract_jsons_spyfall(action)

            if "END_OF_CONVERSATION" in action:
                self._end_condition = "CE"
                self._is_terminal = True
                print(f"There was a chat error.")
                print(action)
                timestep = TimeStep(
                    observation=self.get_observation(),
                    reward=self.get_zero_rewards(),
                    terminal=self._is_terminal,
                )
                return timestep  # stop early to avoid json error
            elif self._prompts[self._prompt_config_mode]["response_format"] == "json":
                if len(json_list) != 1:
                    self._end_condition = "EE"
                    self._is_terminal = True
                    print(f"Player output {action} is not a valid json.")
                    timestep = TimeStep(
                        observation=self.get_observation(),
                        reward=self.get_zero_rewards(),
                        terminal=self._is_terminal,
                    )
                    return timestep  # stop early to avoid json error

            word, arguments = self._get_word_and_argument(
                action=action,
                json_list=json_list,
                response_format=self._prompts[self._prompt_config_mode][
                    "response_format"
                ],
            )

            message = Message(
                agent_name=player_name,
                content=arguments,
                word=word,
                turn=self._current_turn,
            )
            self.message_pool.append_message(message)

            print()

            # Update the counters
            self._current_turn += 1
            if self._next_player_idx < len(self.player_names) - 1:
                self._next_player_idx += 1
            else:
                self._next_player_idx = 0
                self._current_phase = "accuse"
                self._moderator_speak(
                    self._prompts[self._prompt_config_mode]["voting_phase"]
                )
                self._current_turn += 1

            timestep = TimeStep(
                observation=self.get_observation(),
                reward=self.get_zero_rewards(),
                terminal=False,
            )  # Return all the messages

        elif self._current_phase == "accuse":

            json_list = []
            if self._prompts[self._prompt_config_mode]["response_format"] == "json":
                json_list = extract_jsons_spyfall(action)

            # print("Accuse action:", action)

            if "END_OF_CONVERSATION" in action:
                self._end_condition = "CE"
                self._is_terminal = True
                print(f"There was a chat error.")
                timestep = TimeStep(
                    observation=self.get_observation(),
                    reward=self.get_zero_rewards(),
                    terminal=self._is_terminal,
                )
                return timestep  # stop early to avoid json error

            elif self._prompts[self._prompt_config_mode]["response_format"] == "json":
                if len(json_list) != 1:
                    self._end_condition = "EE"
                    self._is_terminal = True

                    print(f"Player output {action} is not a valid json.")
                    timestep = TimeStep(
                        observation=self.get_observation(),
                        reward=self.get_zero_rewards(),
                        terminal=self._is_terminal,
                    )
                    return timestep  # stop early to avoid json error

            word, arguments = self._get_word_and_argument(
                action=action,
                json_list=json_list,
                response_format=self._prompts[self._prompt_config_mode][
                    "response_format"
                ],
            )

            message = Message(
                agent_name=player_name,
                content=arguments,
                word=word,
                turn=self._current_turn,
            )


            if self._prompts[self._prompt_config_mode]["response_format"] == "string":
                self.message_pool.append_message(message)
                pattern = r'\*([^*]+)\*'
                vote = str(re.findall(pattern, arguments))
                vote = self._text2vote(vote)


            if vote in self.player_names:
                self._players_votes[vote] += 1

            if self._next_player_idx < len(self.player_names) - 1:
                self._next_player_idx += 1
                rewards = self.get_zero_rewards()
                terminal = False
            else:

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
                            # Even player votes
                            self._prompts[self._prompt_config_mode][
                                "moderator_says_even_votes"
                            ]
                        )

                    else:
                        # Players vote for the wrong spy
                        self._moderator_speak(
                            self._prompts[self._prompt_config_mode][
                                "moderator_says_wrong_guess"
                            ].format(
                                player_votes=self._players_votes,
                                most_voted_player=max_vote_player,
                            )
                        )
                        # remove the max_vote_player
                        # print(f"Removing player {max_vote_player}"

                        self.player_names.remove(max_vote_player)

                    # change the game self._current_phase = "give clues"
                    if not self._ONE_ROUND:
                        self._current_phase = "give clues"
                        self._players_votes = {name: 0 for name in self.player_names}

                        self._next_player_idx = 0

                        ## Moderator says that it's describing phase again
                        self._moderator_speak(
                            self._prompts[self._prompt_config_mode]["describing_phase"]
                            .format(player=self.player_names[0])
                            .replace(r"{{", "{")
                            .replace(r"}}", "}")
                        )

                        if len(self.player_names) < 4:
                            # Moderator says that spy wins
                            self._moderator_speak(
                                self._prompts[self._prompt_config_mode][
                                    "moderator_says_spy_wins"
                                ]
                            )
                            rewards = self.get_rewards(spy_win=True)
                            self._end_condition = "SPYWINS"  # spy wins
                            terminal = True
                    else:
                        terminal = True
                        self._end_condition = "SPYWINS"
                else:
                    self._end_condition = "SPYLOOSES"  # SW
                    self._moderator_speak(
                        self._prompts[self._prompt_config_mode][
                            "moderator_says_villagers_win"
                        ].format(spy_name=self.spy_name)
                    )
                    terminal = True

                self._current_turn += 1

            timestep = TimeStep(
                observation=self.get_observation(), reward=rewards, terminal=terminal
            )

        # Check if the player signals the end of the conversation
        if self.is_terminal():
            timestep.terminal = True

        return timestep
