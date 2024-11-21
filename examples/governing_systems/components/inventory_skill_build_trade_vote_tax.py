# Modified by Aslan Satary Dizaji, Copyright (c) 2024.

# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""A component to represent each agent's inventory, skill, build, trade, vote, and tax."""

from collections.abc import Callable, Sequence
import concurrent
import dataclasses
import datetime

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
from concordia.utils import helper_functions
import numpy as np
import termcolor


_DEFAULT_QUANTITY = 0


@dataclasses.dataclass(frozen=True)
class ItemTypeConfig:
  """Class for configuring a type of item to track in an Inventory."""

  name: str
  minimum: float = -np.inf
  maximum: float = np.inf
  force_integer: bool = False


class InventorySkillBuildTradeVoteTax(component.Component):
  """A grounded inventory, skill, build, trade, vote, and tax tracking of agents in python."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      item_type_configs: Sequence[ItemTypeConfig],
      player_initial_endowments: dict[str, dict[str, float]],
      clock_now: Callable[[], datetime.datetime],
      government_type: str = 'Full-Libertarian',
      social_reward_function: str = 'Productivity',
      name: str = 'Inventory_Skill_Build_Trade_Vote_Tax',
      verbose: bool = False,
  ):
    """Initialize a grounded inventory component tracking objects in python.

    Args:
      model: a language model.
      memory: an associative memory.
      item_type_configs: sequence of item type configurations.
      player_initial_endowments: dict mapping player name to a dictionary with
        item types as keys and initial endownments as values.
      clock_now: Function to call to get current time.
      government_type: Type of the central social planner which the game master
        plays its role. It could be one of the following three types:
        Full-Libertarian, Full-Utilitarian, and Semi-Libertarian/Utilitarian.
      social_reward_function: Type of the reward function which the central social
        planner considers as a guiding function in determining the due taxes for
        the agents. It could be one of the following two types: Productivity and
        Equality.
      name: the name of this component e.g. Possessions, Account, Property, etc.
      verbose: whether to print the full update chain of thought or not.
    """

    self._model = model
    self._memory = memory
    self._player_initial_endowments = player_initial_endowments
    self._clock_now = clock_now
    self._government_type = government_type
    self._social_reward_function = social_reward_function
    self._name = name
    self._verbose = verbose

    self._item_types = [config.name for config in item_type_configs]
    self._item_types_dict = {
        config.name: config for config in item_type_configs
    }
    self._player_names = list(player_initial_endowments.keys())

    self._inventories = {}
    for player_name, endowment in player_initial_endowments.items():
      self._inventories[player_name] = {
          item_type: endowment.get(item_type, _DEFAULT_QUANTITY)
          for item_type in self._item_types
      }

    self._rank = {}
    if self._government_type == 'Full-Libertarian' or self._government_type == 'Semi-Libertarian/Utilitarian':
      for player in self._player_names:
        self._rank[player] = [2, 1, 0]
    elif self._government_type == 'Full-Utilitarian':
      self._rank = [2, 1, 0]

    self._tax = {}
    for player in self._player_names:
      self._tax[player] = 0

    self._three_types_of_income = {}
    self._three_types_of_income['house trade'] = 0
    self._three_types_of_income['house build'] = 0
    self._three_types_of_income['skill trade'] = 0

    self._borda_vote_count = [0, 0, 0]

    self._history = []
    self._state = ''
    self._partial_states = {name: '' for name in self._player_names}

    # Determine if each item type is a count noun or a mass noun.
    self._is_count_noun = {}

    def check_if_count_noun(item_type):
      self._is_count_noun[item_type] = helper_functions.is_count_noun(
          item_type, self._model
      )
      return

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(self._item_types)
    ) as executor:
      executor.map(check_if_count_noun, self._item_types)

    # Set the initial state's string representation.
    self.update()

  def name(self) -> str:
    """Returns the name of this component."""
    return self._name

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def get_history(self):
    return self._history.copy()

  def _get_player_inventory_str(self, player_name: str) -> str:
    return f"{player_name}'s Inventories: " + str(
        self._inventories[player_name]
    )

  def state(self) -> str:
    return self._state

  def partial_state(
      self,
      player_name: str,
  ) -> str:
    """Return a player-specific view of the component's state."""
    return self._partial_states[player_name]

  def update(self) -> None:
    self._state = '\n'.join([self._get_player_inventory_str(name) for name in self._player_names])
    self._state = '\n'.join(self._three_types_of_income)

    self._partial_states = {name: self._get_player_inventory_str(name) for name in self._player_names}

  def update_after_event(
      self,
      event_statement: str,
  ) -> None:

    all_items = {'money', 'wood', 'stone', 'iron',
                 'red house', 'blue house', 'green house',
                 'red house building skill', 'blue house building skill', 'green house building skill'}

    inventory_change = {}
    for player in self._player_names:
      inventory_change[player] = {}
      for item_type in all_items:
        inventory_change[player][item_type] = 0

    inventory_effects = []
    vote_effects = []
    tax_effects = []

    resource_items = {'wood', 'stone', 'iron'}
    house_items = {'red house', 'blue house', 'green house'}
    skill_items = {'red house building skill', 'blue house building skill', 'green house building skill'}

    chain_of_thought = interactive_document.InteractiveDocument(self._model)

    chain_of_thought.statement(f'List of individuals: {self._player_names}')
    chain_of_thought.statement(f'List of item types: {resource_items}')
    chain_of_thought.statement(f'Event: {event_statement}')

    for player in self._player_names:
      for item_type in resource_items:

        yes_no_answer = chain_of_thought.yes_no_question(
            question=(
                f'In the above transcript, did {player} sell or buy {item_type}(s)? '
                + 'Make sure to take into account items equivalent to the resource items. '
                + 'For example, you may consider "tree" as equivalent to "wood". '
                + 'As another example, you may consider "rock" as equivalent to "stone". '
                + 'As a final example, you may consider "metal" as equivalent to "iron".'
            )
        )

        if yes_no_answer:
          how_many = chain_of_thought.open_question(
              question=(
                  f'How many {item_type}s did {player} sell or buy? '
                  + f'If {player} sold {item_type}(s), the numebr that you '
                  + f'report has to be a negative integer. If {player} bought '
                  + f'{item_type}(s), the number that you report has to be a '
                  + 'positive integer. If you cannot respond, simply respond '
                  + 'with value 1.'
              )
          )

          how_much = chain_of_thought.open_question(
              question=(
                  f'How much did money exchang due to the exchange of {item_type}(s)? '
                  + 'Your response has to a be float number greater than 0. '
                  + 'if there is no mention of prices or exchanged money, simply '
                  + 'respond with value 2.5.'
              )
          )

          if isinstance(how_many, int):
            pass
          else:
            how_many = 1

          if isinstance(how_much, float):
            how_much = abs(how_much)
          else:
            how_much = 2.5

          if how_many == 0:
            how_many = 1

          if how_much == 0 or how_much > 25:
            how_much = 2.5

          if how_many < 0 and abs(how_many) > self._inventories[player][item_type]:
            how_many = -self._inventories[player][item_type]

          elif how_many > 0 and how_much > self._inventories[player]['money']:
            how_much = self._inventories[player]['money']

          prefix = f"[effect on {player}'s Inventory]"

          if how_many > 0:
            self._inventories[player][item_type] = self._inventories[player][item_type] + how_many
            effect = f'{prefix} gained {how_many} {item_type}'
            inventory_effects.append(effect)

            if self._verbose:
              print(termcolor.colored(effect, 'yellow'))

            self._inventories[player]['money'] = self._inventories[player]['money'] - how_much
            effect = f'{prefix} lost {how_much} money'
            inventory_effects.append(effect)

            if self._verbose:
              print(termcolor.colored(effect, 'yellow'))

            inventory_change[player][item_type] = inventory_change[player][item_type] + how_many
            inventory_change[player]['money'] = inventory_change[player]['money'] - how_much

          elif how_many < 0:
            self._inventories[player][item_type] = self._inventories[player][item_type] - abs(how_many)
            effect = f'{prefix} lost {abs(how_many)} {item_type}'
            inventory_effects.append(effect)

            if self._verbose:
              print(termcolor.colored(effect, 'yellow'))

            self._inventories[player]['money'] = self._inventories[player]['money'] + how_much
            effect = f'{prefix} gained {how_much} money'
            inventory_effects.append(effect)

            if self._verbose:
              print(termcolor.colored(effect, 'yellow'))

            inventory_change[player][item_type] = inventory_change[player][item_type] + how_many
            inventory_change[player]['money'] = inventory_change[player]['money'] + how_much

    chain_of_thought.statement(f'List of individuals: {self._player_names}')
    chain_of_thought.statement(f'List of item types: {house_items}')
    chain_of_thought.statement(f'Event: {event_statement}')

    for player in self._player_names:
      for item_type in house_items:

        yes_no_answer = chain_of_thought.yes_no_question(
            question=(
                f'In the above transcript, did {player} sell or buy {item_type}(s)? '
                + f'Make sure only consider selling and buying of {item_type} '
                + 'and not its building.'
            )
        )

        if yes_no_answer:
          how_many = chain_of_thought.open_question(
              question=(
                  f'How many {item_type}s did {player} sell or buy? '
                  + f'If {player} sold {item_type}(s), the numebr that you '
                  + f'report has to be a negative integer. If {player} bought '
                  + f'{item_type}(s), the number that you report has to be a '
                  + 'positive integer. If you cannot respond, simply respond '
                  + 'with value 1.'
              )
          )

          how_much = chain_of_thought.open_question(
              question=(
                  f'How much did money exchang due to the exchange of {item_type}(s)? '
                  + 'Your response has to a be float number greater than 0. '
                  + 'if there is no mention of prices or exchanged money, simply '
                  + 'respond with value 15.'
              )
          )

          if isinstance(how_many, int):
            pass
          else:
            how_many = 1

          if isinstance(how_much, float):
            how_much = abs(how_much)
          else:
            how_much = 15

          if how_many == 0:
            how_many = 1

          if how_much == 0 or how_much > 150:
            how_much = 15

          if how_many < 0 and abs(how_many) > self._inventories[player][item_type]:
            how_many = -self._inventories[player][item_type]

          elif how_many > 0 and how_much > self._inventories[player]['money']:
            how_much = self._inventories[player]['money']

          prefix = f"[effect on {player}'s Inventory]"

          if how_many > 0:
            self._inventories[player][item_type] = self._inventories[player][item_type] + how_many
            effect = f'{prefix} gained {how_many} {item_type}'
            inventory_effects.append(effect)

            if self._verbose:
              print(termcolor.colored(effect, 'yellow'))

            self._inventories[player]['money'] = self._inventories[player]['money'] - how_much
            effect = f'{prefix} lost {how_much} money'
            inventory_effects.append(effect)

            if self._verbose:
              print(termcolor.colored(effect, 'yellow'))

            self._three_types_of_income['house trade'] = self._three_types_of_income['house trade'] + how_much

            inventory_change[player][item_type] = inventory_change[player][item_type] + how_many
            inventory_change[player]['money'] = inventory_change[player]['money'] - how_much

          elif how_many < 0:
            self._inventories[player][item_type] = self._inventories[player][item_type] - abs(how_many)
            effect = f'{prefix} lost {abs(how_many)} {item_type}'
            inventory_effects.append(effect)

            if self._verbose:
              print(termcolor.colored(effect, 'yellow'))

            self._inventories[player]['money'] = self._inventories[player]['money'] + how_much
            effect = f'{prefix} gained {abs(how_much)} money'
            inventory_effects.append(effect)

            if self._verbose:
              print(termcolor.colored(effect, 'yellow'))

            inventory_change[player][item_type] = inventory_change[player][item_type] + how_many
            inventory_change[player]['money'] = inventory_change[player]['money'] + how_much

    chain_of_thought.statement(f'List of individuals: {self._player_names}')
    chain_of_thought.statement(f'List of item types: {house_items}')
    chain_of_thought.statement(f'Event: {event_statement}')

    for player in self._player_names:
      for item_type in house_items:

        yes_no_answer = chain_of_thought.yes_no_question(
            question=(
                f'In the above transcript, did {player} build {item_type}(s)? '
                + 'Make sure only consider building houses and not selling '
                + 'or buying them.'
            )
        )

        if yes_no_answer:
          how_many = chain_of_thought.open_question(
              question=(
                  f'How many {item_type}s did {player} build? '
                  + 'Make sure only consider building houses and not selling '
                  + 'or buying them. The numebr that you report has to be an '
                  + 'integer greater than 0. If you cannot respond, simply '
                  + 'respond with value 1.'
              )
          )

          if isinstance(how_many, int):
            how_many = abs(how_many)
          else:
            how_many = 1

          if how_many == 0:
            how_many = 1

          prefix = f"[effect on {player}'s Inventory]"

          if item_type == 'red house':
            if self._inventories[player]['red house building skill'] <= 5:
              how_many = 0
            if self._inventories[player]['wood'] < how_many or self._inventories[player]['stone'] < how_many:
              how_many = min(self._inventories[player]['wood'], self._inventories[player]['stone'])

            if how_many > 0:
              self._inventories[player]['wood'] = self._inventories[player]['wood'] - how_many
              effect = f'{prefix} lost {how_many} woods'
              inventory_effects.append(effect)

              if self._verbose:
                print(termcolor.colored(effect, 'yellow'))

              self._inventories[player]['stone'] = self._inventories[player]['stone'] - how_many
              effect = f'{prefix} lost {how_many} stones'
              inventory_effects.append(effect)

              if self._verbose:
                print(termcolor.colored(effect, 'yellow'))

              effect = f'{prefix} gained {how_many} red houses'
              self._inventories[player]['red house'] = self._inventories[player]['red house'] + how_many
              inventory_effects.append(effect)

              if self._verbose:
                print(termcolor.colored(effect, 'yellow'))

              amount = how_many * self._inventories[player]['red house building skill']

              self._inventories[player]['money'] = self._inventories[player]['money'] + amount
              effect = f'{prefix} gained {amount} money'
              inventory_effects.append(effect)

              if self._verbose:
                print(termcolor.colored(effect, 'yellow'))

              self._three_types_of_income['house build'] = self._three_types_of_income['house build'] + amount

              inventory_change[player]['wood'] = inventory_change[player]['wood'] - how_many
              inventory_change[player]['stone'] = inventory_change[player]['stone'] - how_many
              inventory_change[player]['red house'] = inventory_change[player]['red house'] + how_many
              inventory_change[player]['money'] = inventory_change[player]['money'] + amount

          elif item_type == 'blue house':
            if self._inventories[player]['blue house building skill'] <= 5:
              how_many = 0
            if self._inventories[player]['wood'] < how_many or self._inventories[player]['iron'] < how_many:
              how_many = min(self._inventories[player]['wood'], self._inventories[player]['iron'])

            if how_many > 0:
              self._inventories[player]['wood'] = self._inventories[player]['wood'] - how_many
              effect = f'{prefix} lost {how_many} woods'
              inventory_effects.append(effect)

              if self._verbose:
                print(termcolor.colored(effect, 'yellow'))

              self._inventories[player]['iron'] = self._inventories[player]['iron'] - how_many
              effect = f'{prefix} lost {how_many} irons'
              inventory_effects.append(effect)

              if self._verbose:
                print(termcolor.colored(effect, 'yellow'))

              effect = f'{prefix} gained {how_many} blue houses'
              self._inventories[player]['blue house'] = self._inventories[player]['blue house'] + how_many
              inventory_effects.append(effect)

              if self._verbose:
                print(termcolor.colored(effect, 'yellow'))

              amount = how_many * self._inventories[player]['blue house building skill']

              self._inventories[player]['money'] = self._inventories[player]['money'] + amount
              effect = f'{prefix} gained {amount} money'
              inventory_effects.append(effect)

              if self._verbose:
                print(termcolor.colored(effect, 'yellow'))

              self._three_types_of_income['house build'] = self._three_types_of_income['house build'] + amount

              inventory_change[player]['wood'] = inventory_change[player]['wood'] - how_many
              inventory_change[player]['iron'] = inventory_change[player]['iron'] - how_many
              inventory_change[player]['blue house'] = inventory_change[player]['blue house'] + how_many
              inventory_change[player]['money'] = inventory_change[player]['money'] + amount

          elif item_type == 'green house':
            if self._inventories[player]['green house building skill'] <= 5:
              how_many = 0
            if self._inventories[player]['stone'] < how_many or self._inventories[player]['iron'] < how_many:
              how_many = min(self._inventories[player]['stone'], self._inventories[player]['iron'])

            if how_many > 0:
              self._inventories[player]['stone'] = self._inventories[player]['stone'] - how_many
              effect = f'{prefix} lost {how_many} stones'
              inventory_effects.append(effect)

              if self._verbose:
                print(termcolor.colored(effect, 'yellow'))

              self._inventories[player]['iron'] = self._inventories[player]['iron'] - how_many
              effect = f'{prefix} lost {how_many} irons'
              inventory_effects.append(effect)

              if self._verbose:
                print(termcolor.colored(effect, 'yellow'))

              effect = f'{prefix} gained {how_many} green houses'
              self._inventories[player]['green house'] = self._inventories[player]['green house'] + how_many
              inventory_effects.append(effect)

              if self._verbose:
                print(termcolor.colored(effect, 'yellow'))

              amount = how_many * self._inventories[player]['green house building skill']

              self._inventories[player]['money'] = self._inventories[player]['money'] + amount
              effect = f'{prefix} gained {amount} money'
              inventory_effects.append(effect)

              if self._verbose:
                print(termcolor.colored(effect, 'yellow'))

              self._three_types_of_income['house build'] = self._three_types_of_income['house build'] + amount

              inventory_change[player]['stone'] = inventory_change[player]['stone'] - how_many
              inventory_change[player]['iron'] = inventory_change[player]['iron'] - how_many
              inventory_change[player]['green house'] = inventory_change[player]['green house'] + how_many
              inventory_change[player]['money'] = inventory_change[player]['money'] + amount

    chain_of_thought.statement(f'List of individuals: {self._player_names}')
    chain_of_thought.statement(f'List of item types: {skill_items}')
    chain_of_thought.statement(f'Event: {event_statement}')

    for player in self._player_names:
      for item_type in skill_items:

        yes_no_answer = chain_of_thought.yes_no_question(
            question=(
                f'In the above transcript, did {player} sell or buy {item_type}(s)? '
                + 'Make sure only consider buying and selling of house building '
                + 'skills and not buying, selling, or building of different house '
                + 'types.'
            )
        )

        if yes_no_answer:
          how_many = chain_of_thought.open_question(
              question=(
                  f'How many {item_type}s did {player} sell or buy? '
                  + 'Make sure only consider buying and selling of house building '
                  + 'skills and not buying, selling, or building of different house '
                  + f'types. If {player} sold {item_type}(s), the numebr that you '
                  + f'report has to be a negative integer. If {player} bought '
                  + f'{item_type}(s), the number that you report has to be a '
                  + 'positive integer. If you cannot respond, simply respond '
                  + 'with value 1.'
              )
          )

          how_much = chain_of_thought.open_question(
              question=(
                  f'How much did money exchang due to the exchange of {item_type}(s)? '
                  + 'Your response has to a be float number greater than 0. '
                  + 'if there is no mention of prices or exchanged money, simply '
                  + 'respond with value 1.5.'
              )
          )

          if isinstance(how_many, int):
            pass
          else:
            how_many = 1

          if isinstance(how_much, float):
            how_much = abs(how_much)
          else:
            how_much = 1.5

          if how_many == 0:
            how_many = 1

          if how_much == 0 or how_much > 15:
            how_much = 1.5

          if how_many > 0 and how_much > self._inventories[player]['money']:
            how_much = self._inventories[player]['money']

          prefix = f"[effect on {player}'s Inventory]"

          if how_many > 0:
            self._inventories[player][item_type] = self._inventories[player][item_type] + how_many
            effect = f'{prefix} gained {how_many} {item_type}'
            inventory_effects.append(effect)

            if self._verbose:
              print(termcolor.colored(effect, 'yellow'))

            self._inventories[player]['money'] = self._inventories[player]['money'] - how_much
            effect = f'{prefix} lost {how_much} money'
            inventory_effects.append(effect)

            if self._verbose:
              print(termcolor.colored(effect, 'yellow'))

            self._three_types_of_income['skill trade'] = self._three_types_of_income['skill trade'] + how_much

            inventory_change[player][item_type] = inventory_change[player][item_type] + how_many
            inventory_change[player]['money'] = inventory_change[player]['money'] - how_much

          elif how_many < 0:
            self._inventories[player]['money'] = self._inventories[player]['money'] + how_much
            effect = f'{prefix} gained {how_much} money'
            inventory_effects.append(effect)

            if self._verbose:
              print(termcolor.colored(effect, 'yellow'))

            inventory_change[player][item_type] = inventory_change[player][item_type] + how_many
            inventory_change[player]['money'] = inventory_change[player]['money'] + how_much

    if self._government_type == "Full-Libertarian" or self._government_type == "Semi-Libertarian/Utilitarian":
      chain_of_thought.statement(f'Event: {event_statement}')

      proceed = chain_of_thought.yes_no_question(
          question=(
              'In the above transcript, did any of the listed individuals vote or '
              'rank three resources (wood, stone, and iron). Please consider even '
              'those actions which imply voting or ranking such as prefering one '
              'item to other ones.'
          )
      )

      if proceed:
        players_who_voted_str = chain_of_thought.open_question(
            question=(
                'Which individuals did vote or rank the resouces?\n'
                + 'Respond with a comma-separated list, for example: \n'
                + 'Jacob,Alfred,Patricia'
            )
        )

        players_who_voted = players_who_voted_str.split(',')

        for player in players_who_voted:

          player = player.strip(' ')

          if player in self._player_names:

            vote = chain_of_thought.open_question(
                question=(
                    f'How did {player} vote or rank the three resources? '
                    + 'Make sure you report voting or ranking with the '
                    + 'comma-separated list. For example, if the individual '
                    + 'prefers stone to wood and wood to iron, then the final '
                    + 'ranking is like this: stone,wood,iron.'
                )
            )

            flag = True

            vote = vote.split(',')

            if len(vote) != 3:
              flag = False

            else:
              if vote[0].strip(' ') == 'wood':
                self._rank[player][0] = 2
              elif vote[1].strip(' ') == 'wood':
                self._rank[player][0] = 1
              elif vote[2].strip(' ') == 'wood':
                self._rank[player][0] == 0

              if vote[0].strip(' ') == 'stone':
                self._rank[player][1] = 2
              elif vote[1].strip(' ') == 'stone':
                self._rank[player][1] = 1
              elif vote[2].strip(' ') == 'stone':
                self._rank[player][1] == 0

              if vote[0].strip(' ') == 'iron':
                self._rank[player][2] = 2
              elif vote[1].strip(' ') == 'iron':
                self._rank[player][2] = 1
              elif vote[2].strip(' ') == 'iron':
                self._rank[player][2] == 0

            if flag == False:
              if inventory_change[player]['wood'] >= inventory_change[player]['stone'] and \
                inventory_change[player]['wood'] >= inventory_change[player]['iron'] and \
                inventory_change[player]['stone'] >= inventory_change[player]['iron']:

                self._rank[player] = [2, 1, 0]
              elif inventory_change[player]['wood'] >= inventory_change[player]['stone'] and \
                inventory_change[player]['wood'] >= inventory_change[player]['iron'] and \
                inventory_change[player]['iron'] >= inventory_change[player]['stone']:

                self._rank[player] = [2, 0, 1]
              elif inventory_change[player]['stone'] >= inventory_change[player]['wood'] and \
                inventory_change[player]['stone'] >= inventory_change[player]['iron'] and \
                inventory_change[player]['wood'] >= inventory_change[player]['iron']:

                self._rank[player] = [1, 2, 0]
              elif inventory_change[player]['iron'] >= inventory_change[player]['wood'] and \
                inventory_change[player]['iron'] >= inventory_change[player]['stone'] and \
                inventory_change[player]['wood'] >= inventory_change[player]['stone']:

                self._rank[player] = [1, 0, 2]
              elif inventory_change[player]['stone'] >= inventory_change[player]['wood'] and \
                inventory_change[player]['stone'] >= inventory_change[player]['iron'] and \
                inventory_change[player]['iron'] >= inventory_change[player]['wood']:

                self._rank[player] = [0, 2, 1]
              elif inventory_change[player]['iron'] >= inventory_change[player]['wood'] and \
                inventory_change[player]['iron'] >= inventory_change[player]['stone'] and \
                inventory_change[player]['stone'] >= inventory_change[player]['wood']:

               self._rank[player] = [0, 1, 2]

            prefix = f"[effect on {player}'s Vote]"

            if self._rank[player] == [2, 1, 0]:
              effect = f'{prefix} voted wood, stone, iron'
            if self._rank[player] == [2, 0, 1]:
              effect = f'{prefix} voted wood, iron, stone'
            if self._rank[player] == [1, 2, 0]:
              effect = f'{prefix} voted stone, wood, iron'
            if self._rank[player] == [1, 0, 2]:
              effect = f'{prefix} voted iron, wood, stone'
            if self._rank[player] == [0, 2, 1]:
              effect = f'{prefix} voted stone, iron, wood'
            if self._rank[player] == [0, 1, 2]:
              effect = f'{prefix} voted iron, stone, wood'

            vote_effects.append(effect)

            if self._verbose:
              print(termcolor.colored(effect, 'yellow'))

        for player in self._player_names:
          if player not in players_who_voted:

            if inventory_change[player]['wood'] >= inventory_change[player]['stone'] and \
              inventory_change[player]['wood'] >= inventory_change[player]['iron'] and \
              inventory_change[player]['stone'] >= inventory_change[player]['iron']:

              self._rank[player] = [2, 1, 0]
            elif inventory_change[player]['wood'] >= inventory_change[player]['stone'] and \
              inventory_change[player]['wood'] >= inventory_change[player]['iron'] and \
              inventory_change[player]['iron'] >= inventory_change[player]['stone']:

              self._rank[player] = [2, 0, 1]
            elif inventory_change[player]['stone'] >= inventory_change[player]['wood'] and \
              inventory_change[player]['stone'] >= inventory_change[player]['iron'] and \
              inventory_change[player]['wood'] >= inventory_change[player]['iron']:

              self._rank[player] = [1, 2, 0]
            elif inventory_change[player]['iron'] >= inventory_change[player]['wood'] and \
              inventory_change[player]['iron'] >= inventory_change[player]['stone'] and \
              inventory_change[player]['wood'] >= inventory_change[player]['stone']:

              self._rank[player] = [1, 0, 2]
            elif inventory_change[player]['stone'] >= inventory_change[player]['wood'] and \
              inventory_change[player]['stone'] >= inventory_change[player]['iron'] and \
              inventory_change[player]['iron'] >= inventory_change[player]['wood']:

              self._rank[player] = [0, 2, 1]
            elif inventory_change[player]['iron'] >= inventory_change[player]['wood'] and \
              inventory_change[player]['iron'] >= inventory_change[player]['stone'] and \
              inventory_change[player]['stone'] >= inventory_change[player]['wood']:

              self._rank[player] = [0, 1, 2]

            prefix = f"[effect on {player}'s Vote]"

            if self._rank[player] == [2, 1, 0]:
              effect = f'{prefix} voted wood, stone, iron'
            if self._rank[player] == [2, 0, 1]:
              effect = f'{prefix} voted wood, iron, stone'
            if self._rank[player] == [1, 2, 0]:
              effect = f'{prefix} voted stone, wood, iron'
            if self._rank[player] == [1, 0, 2]:
              effect = f'{prefix} voted iron, wood, stone'
            if self._rank[player] == [0, 2, 1]:
              effect = f'{prefix} voted stone, iron, wood'
            if self._rank[player] == [0, 1, 2]:
              effect = f'{prefix} voted iron, stone, wood'

            vote_effects.append(effect)

            if self._verbose:
              print(termcolor.colored(effect, 'yellow'))

      elif proceed == False:
        for player in self._player_names:

          if inventory_change[player]['wood'] >= inventory_change[player]['stone'] and \
            inventory_change[player]['wood'] >= inventory_change[player]['iron'] and \
            inventory_change[player]['stone'] >= inventory_change[player]['iron']:

            self._rank[player] = [2, 1, 0]
          elif inventory_change[player]['wood'] >= inventory_change[player]['stone'] and \
            inventory_change[player]['wood'] >= inventory_change[player]['iron'] and \
            inventory_change[player]['iron'] >= inventory_change[player]['stone']:

            self._rank[player] = [2, 0, 1]
          elif inventory_change[player]['stone'] >= inventory_change[player]['wood'] and \
            inventory_change[player]['stone'] >= inventory_change[player]['iron'] and \
            inventory_change[player]['wood'] >= inventory_change[player]['iron']:

            self._rank[player] = [1, 2, 0]
          elif inventory_change[player]['iron'] >= inventory_change[player]['wood'] and \
            inventory_change[player]['iron'] >= inventory_change[player]['stone'] and \
            inventory_change[player]['wood'] >= inventory_change[player]['stone']:

            self._rank[player] = [1, 0, 2]
          elif inventory_change[player]['stone'] >= inventory_change[player]['wood'] and \
            inventory_change[player]['stone'] >= inventory_change[player]['iron'] and \
            inventory_change[player]['iron'] >= inventory_change[player]['wood']:

            self._rank[player] = [0, 2, 1]
          elif inventory_change[player]['iron'] >= inventory_change[player]['wood'] and \
            inventory_change[player]['iron'] >= inventory_change[player]['stone'] and \
            inventory_change[player]['stone'] >= inventory_change[player]['wood']:

            self._rank[player] = [0, 1, 2]

          prefix = f"[effect on {player}'s Vote]"

          if self._rank[player] == [2, 1, 0]:
            effect = f'{prefix} voted wood, stone, iron'
          if self._rank[player] == [2, 0, 1]:
            effect = f'{prefix} voted wood, iron, stone'
          if self._rank[player] == [1, 2, 0]:
            effect = f'{prefix} voted stone, wood, iron'
          if self._rank[player] == [1, 0, 2]:
            effect = f'{prefix} voted iron, wood, stone'
          if self._rank[player] == [0, 2, 1]:
            effect = f'{prefix} voted stone, iron, wood'
          if self._rank[player] == [0, 1, 2]:
            effect = f'{prefix} voted iron, stone, wood'

          vote_effects.append(effect)

          if self._verbose:
            print(termcolor.colored(effect, 'yellow'))

      for player in self._player_names:
        self._borda_vote_count = np.array(self._borda_vote_count) + np.array(self._rank[player])

    elif self._government_type == "Full-Utilitarian":
      resource_items = {'money', 'wood', 'stone', 'iron'}

      for player in self._player_names:
        for item in resource_items:
          chain_of_thought.statement(f'The inventory of {item} of {player} changed as the following: {inventory_change[player][item]}')

      vote = chain_of_thought.open_question(
          question=(
              'You are the central social planner or the government of this city '
              'comprised of the above individuals. These individuals performed '
              'the above economic activities, and gained or lost some amount of '
              'money or resources (wood, stone, iron). As a utilitarian social '
              'planner who cares about all individuals of the city equally and '
              'want them to prosper, how do you vote or rank these resources so '
              'your vote or ranking reflects the desire of these individuals. '
              'Think deeply and try to come up with a vote or rank even if you '
              'are not confident. Respond with a comma-separated list, '
              'for example: wood,stone,iron.'
          )
      )

      flag = True

      vote = vote.split(',')

      if len(vote) != 3:
        flag = False

      else:
        if vote[0].strip(' ') == 'wood':
          self._rank[0] = 2
        elif vote[1].strip(' ') == 'wood':
          self._rank[0] = 1
        elif vote[2].strip(' ') == 'wood':
          self._rank[0] = 0

        if vote[0].strip(' ') == 'stone':
          self._rank[1] = 2
        elif vote[1].strip(' ') == 'stone':
          self._rank[1] = 1
        elif vote[2].strip(' ') == 'stone':
          self._rank[1] = 0

        if vote[0].strip(' ') == 'iron':
          self._rank[2] = 2
        elif vote[1].strip(' ') == 'iron':
          self._rank[2] = 1
        elif vote[2].strip(' ') == 'iron':
          self._rank[2] = 0

      if flag == False:
        resource_items_prime = {'wood', 'stone', 'iron'}
        inventory_change_all = {}

        for item in resource_items_prime:
          inventory_change_all[item] = 0

          for player in self._player_names:
            inventory_change_all[item] = inventory_change_all[item] + abs(inventory_change[player][item])

        if inventory_change_all['wood'] >= inventory_change_all['stone'] and \
          inventory_change_all['wood'] >= inventory_change_all['iron'] and \
          inventory_change_all['stone'] >= inventory_change_all['iron']:

          self._rank = [2, 1, 0]
        elif inventory_change_all['wood'] >= inventory_change_all['stone'] and \
          inventory_change_all['wood'] >= inventory_change_all['iron'] and \
          inventory_change_all['iron'] >= inventory_change_all['stone']:

          self._rank = [2, 0, 1]
        elif inventory_change_all['stone'] >= inventory_change_all['wood'] and \
          inventory_change_all['stone'] >= inventory_change_all['iron'] and \
          inventory_change_all['wood'] >= inventory_change_all['iron']:

          self._rank = [1, 2, 0]
        elif inventory_change_all['iron'] >= inventory_change_all['wood'] and \
          inventory_change_all['iron'] >= inventory_change_all['stone'] and \
          inventory_change_all['wood'] >= inventory_change_all['stone']:

          self._rank = [1, 0, 2]
        elif inventory_change_all['stone'] >= inventory_change_all['wood'] and \
          inventory_change_all['stone'] >= inventory_change_all['iron'] and \
          inventory_change_all['iron'] >= inventory_change_all['wood']:

          self._rank = [0, 2, 1]
        elif inventory_change_all['iron'] >= inventory_change_all['wood'] and \
          inventory_change_all['iron'] >= inventory_change_all['stone'] and \
          inventory_change_all['stone'] >= inventory_change_all['wood']:

          self._rank = [0, 1, 2]

      prefix = f"[effect on central social planner's Vote]"

      if self._rank == [2, 1, 0]:
        effect = f'{prefix} voted wood, stone, iron'
      if self._rank == [2, 0, 1]:
        effect = f'{prefix} voted wood, iron, stone'
      if self._rank == [1, 2, 0]:
        effect = f'{prefix} voted stone, wood, iron'
      if self._rank == [1, 0, 2]:
        effect = f'{prefix} voted iron, wood, stone'
      if self._rank == [0, 2, 1]:
        effect = f'{prefix} voted stone, iron, wood'
      if self._rank == [0, 1, 2]:
        effect = f'{prefix} voted iron, stone, wood'

      vote_effects.append(effect)

      if self._verbose:
          print(termcolor.colored(effect, 'yellow'))

      self._borda_vote_count = np.array(self._rank)

    if self._social_reward_function == "Productivity":
      for player_1 in self._player_names:
        for player_2 in self._player_names:
          inventory_change_money = inventory_change[player_2]['money']
          chain_of_thought.statement(f'The inventory of money of {player_2} changed as the following: {inventory_change_money}')

        flag = True
        total_tax = 0
        inventory_change_money = inventory_change[player_1]['money']

        amount = chain_of_thought.open_question(
            question=(
                'You are the central social planner or the government of this city. '
                'You are responsible to set the due tax of each individual of this '
                'city considering her or his income in comparison to income of others. '
                'You treat individuals equally meaning that the only factor which '
                'you consider is her or his income in comparison to others. '
                'Moreover, you care about the overall productivity of the people of '
                'this city, knowing that if you set a high due tax for an individual '
                'who has a high income, you might discourage her or him to work harder '
                'in future. Now, considering all these factors, and the incomes of '
                'all individuals of this city, how much tax would you set for '
                + f'{player_1} who has earned {inventory_change_money}? '
                'Think deeply and try to come up with a value which has to be less '
                + f'than the actual income. If the income of {player_1} is less than '
                'zero, respond with the value 0. Also, if you are not able to answer '
                'the question, respond with the value 0.'
            )
        )

        if isinstance(amount, float):
          self._tax[player_1] = amount
        else:
          flag = False

        if flag == True and inventory_change_money < amount:
          self._tax[player_1] = inventory_change_money
        elif inventory_change_money < 0:
          self._tax[player_1] = 0
        elif (amount == 0 and inventory_change_money > 0) or (flag == False and inventory_change_money > 0):
          self._tax[player_1] = 0.3 * inventory_change_money
          self._inventories[player_1]['money'] = self._inventories[player_1]['money'] - self._tax[player_1]

        prefix = f"[effect on {player_1}'s Tax]"
        effect = f'{prefix} is equal to {self._tax[player_1]}'
        tax_effects.append(effect)

        if self._verbose:
          print(termcolor.colored(effect, 'yellow'))

      for player in self._player_names:
        total_tax = total_tax + self._tax[player]

      if self._government_type == 'Full-Libertarian':
        for player in self._player_names:
          self._inventories[player]['wood'] = self._inventories[player]['wood'] + int(self._tax[player] * self._rank[player][0] / 3)
          self._inventories[player]['stone'] = self._inventories[player]['stone'] + int(self._tax[player] * self._rank[player][1] / 3)
          self._inventories[player]['iron'] = self._inventories[player]['iron'] + int(self._tax[player] * self._rank[player][2] / 3)
      if self._government_type == 'Semi-Libertarian/Utilitarian':
        for player in self._player_names:
          self._inventories[player]['wood'] = self._inventories[player]['wood'] + int(total_tax / 6 * self._borda_vote_count[0] / sum(self._borda_vote_count))
          self._inventories[player]['stone'] = self._inventories[player]['stone'] + int(total_tax / 6 * self._borda_vote_count[1] / sum(self._borda_vote_count))
          self._inventories[player]['iron'] = self._inventories[player]['iron'] + int(total_tax / 6 * self._borda_vote_count[2] / sum(self._borda_vote_count))
      if self._government_type == 'Full-Utilitarian':
        for player in self._player_names:
          self._inventories[player]['wood'] = self._inventories[player]['wood'] + int(total_tax / 6 * self._rank[0] / 3)
          self._inventories[player]['stone'] = self._inventories[player]['stone'] + int(total_tax / 6 * self._rank[1] / 3)
          self._inventories[player]['iron'] = self._inventories[player]['iron'] + int(total_tax / 6 * self._rank[2] / 3)

    elif self._social_reward_function == "Equality":
      for player_1 in self._player_names:
        for player_2 in self._player_names:
          inventory_change_money = inventory_change[player_2]['money']
          chain_of_thought.statement(f'The inventory of money of {player_2} changed as the following: {inventory_change_money}')

        flag = True
        total_tax = 0
        inventory_change_money = inventory_change[player_1]['money']

        amount = chain_of_thought.open_question(
            question=(
                'You are the central social planner or the government of this city. '
                'You are responsible to set the due tax of each individual of this '
                'city considering her or his income in comparison to income of others. '
                'You treat individuals equally meaning that the only factor which '
                'you consider is her or his income in comparison to others. '
                'Moreover, you care about the overall equality of the people of '
                'this city, knowing that if someone earns a higher income in '
                'comparison to others, you should set a high tax for her or him, '
                'so to distribute the wealth as much as evenly among the people '
                'of the city. Now, considering all these factors, and the incomes of '
                'all individuals of this city, how much tax would you set for '
                + f'{player_1} who has earned {inventory_change_money}? '
                'Think deeply and try to come up with a value which has to be less '
                + f'than the actual income. If the income of {player_1} is less than '
                'zero, respond with the value 0. Also, if you are not able to answer '
                'the question, respond with the value 0.'
            )
        )

        if isinstance(amount, float):
          self._tax[player_1] = amount
        else:
          flag = False

        if flag == True and inventory_change_money < amount:
          self._tax[player_1] = inventory_change_money
        elif inventory_change_money < 0:
          self._tax[player_1] = 0
        elif (amount == 0 and inventory_change_money > 0) or (flag == False and inventory_change_money > 0):
          self._tax[player_1] = 0.3 * inventory_change_money
          self._inventories[player_1]['money'] = self._inventories[player_1]['money'] - self._tax[player_1]

        prefix = f"[effect on {player_1}'s Tax]"
        effect = f'{prefix} is equal to {self._tax[player_1]}'
        tax_effects.append(effect)

        if self._verbose:
          print(termcolor.colored(effect, 'yellow'))

      for player in self._player_names:
        total_tax = total_tax + self._tax[player]

      if self._government_type == 'Full-Libertarian':
        for player in self._player_names:
          self._inventories[player]['wood'] = self._inventories[player]['wood'] + int(self._tax[player] * self._rank[player][0] / 3)
          self._inventories[player]['stone'] = self._inventories[player]['stone'] + int(self._tax[player] * self._rank[player][1] / 3)
          self._inventories[player]['iron'] = self._inventories[player]['iron'] + int(self._tax[player] * self._rank[player][2] / 3)
      if self._government_type == 'Semi-Libertarian/Utilitarian':
        for player in self._player_names:
          self._inventories[player]['wood'] = self._inventories[player]['wood'] + int(total_tax / 6 * self._borda_vote_count[0] / sum(self._borda_vote_count))
          self._inventories[player]['stone'] = self._inventories[player]['stone'] + int(total_tax / 6 * self._borda_vote_count[1] / sum(self._borda_vote_count))
          self._inventories[player]['iron'] = self._inventories[player]['iron'] + int(total_tax / 6 * self._borda_vote_count[2] / sum(self._borda_vote_count))
      if self._government_type == 'Full-Utilitarian':
        for player in self._player_names:
          self._inventories[player]['wood'] = self._inventories[player]['wood'] + int(total_tax / 6 * self._rank[0] / 3)
          self._inventories[player]['stone'] = self._inventories[player]['stone'] + int(total_tax / 6 * self._rank[1] / 3)
          self._inventories[player]['iron'] = self._inventories[player]['iron'] + int(total_tax / 6 * self._rank[2] / 3)

    # Update the string representation of all inventories.
    self.update()

    if self._verbose:
      print(termcolor.colored(chain_of_thought.view().text(), 'yellow'))
      print(termcolor.colored(self.state(), 'yellow'))

    update_log = {
        'date': self._clock_now(),
        'Summary': str(self._three_types_of_income),
        'Inventories': str(self._inventories),
        'Vote': str(self._rank),
        'Tax': str(self._tax),
        'Chain of thought': {
            'Summary': f'{self._name} chain of thought',
            'Chain': chain_of_thought.view().text().splitlines(),
        },
    }

    self._memory.extend(inventory_effects)
    self._memory.extend(vote_effects)
    self._memory.extend(tax_effects)

    self._history.append(update_log)
