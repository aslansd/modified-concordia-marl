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

r"""Evaluate the submitted agent on all scenarios.

Usage:
cd {concordia_root}/
python examples/modular/launch_concordia_challenge_evaluation.py \
  --agent=AGENT_NAME \
  --api_type=API_TYPE \
  --model=MODEL_NAME \
  --embedder=EMBEDDER_NAME \
  --num_repetitions_per_scenario=NUM_REPETITIONS_PER_SCENARIO

Where AGENT_NAME indicates a file under concordia/factory/agent,
ENVIRONMENT_NAME indicates a file under examples/modular/environment,
API_TYPE is one of the options named in concordia/language_model/utils.py,
e.g. 'google_aistudio_model', 'openai', 'mistral', 'ollama', 'amazon_bedrock'.
MODEL_NAME is a specific model under the chosen API_TYPE. See the corresponding
wrapper in concordia/language_model/ for the link to the website where the
model names are listed for each type of API.
and EMBEDDER_NAME specifies a sentence transformers embedding model listed at
https://huggingface.co/sentence-transformers.
NUM_REPETITIONS_PER_SCENARIO specifies how many times to repeat each scenario,
averaging the results to produce a single score per scenario.

This script will download the embedder from huggingface and cache it locally.

To debug without spending money on API calls, pass the extra flag:
  --disable_language_model
It replaces the language model with a null model that always returns an empty
string when asked for a free response and always selects the first option when
asked for a multiple choice.

This script will write a json file with the results of the evaluation to the
current working directory. The file will be named
  AGENT_NAME__MODEL_NAME__EMBEDDER_NAME.json
and will contain a list of json-serializable objects, each one containing
results on all scenarios for the selected (agent, model, embedder). For each
scenario, this script also writes an html file with its full text log. The file
will be named
  SCENARIO_NAME__YYYY-MM-DD HH:MM:SS.html
where SCENARIO_NAME is the name of the scenario and the date and time are the
time when the simulation was run.
The script also writes a text file in the current working directory with the
name of each evaluated agent:
  agents__MODEL_NAME__EMBEDDER_NAME.txt
This file is used to keep track of which agents have already been evaluated. For
a given MODEL_NAME and EMBEDDER_NAME. If the selected agent is already in the
list, the script will raise an error.

After running this script you can run `calculate_ratings.py` to compute Elo
ratings. The `calculate_ratings.py` script loads the json files written by this
script and computes the Elo ratings for all agents that were been tested with
the same model and embedder.
"""

import argparse
import datetime
import importlib
import pathlib
import sys

from concordia.language_model import utils
from concordia.typing import logging as logging_lib
from concordia.utils import measurements as measurements_lib
import numpy as np
import sentence_transformers

concordia_root_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.append(f'{concordia_root_dir}')
# pylint: disable=g-import-not-at-top, g-bad-import-order
from examples.modular.scenario import scenarios as scenarios_lib
from examples.modular.utils import files as file_utils

# Setup for command line arguments
parser = argparse.ArgumentParser(
    description='Run a Concordia Challenge evaluation.'
)
parser.add_argument(
    '--agent',
    action='store',
    default='rational_entity_agent__main_role',
    dest='agent_name',
)
parser.add_argument(
    '--api_type', action='store', default='mistral', dest='api_type'
)
parser.add_argument(
    '--model', action='store', default='codestral-latest', dest='model_name'
)
parser.add_argument(
    '--embedder',
    action='store',
    default='all-mpnet-base-v2',
    dest='embedder_name',
)
parser.add_argument(
    '--num_repetitions_per_scenario',
    action='store',
    type=int,
    default=2,
    dest='num_repetitions_per_scenario',
)
parser.add_argument(
    '--disable_language_model',
    action='store_true',
    help=(
        'replace the language model with a null model. This '
        'makes it possible to debug without spending money '
        'on api calls.'
    ),
    default=False,
    dest='disable_language_model',
)
parser.add_argument(
    '--exclude_from_elo_calculation',
    action='store_true',
    help=(
        'Use this option to write and analyze test data. It '
        'will be automatically enabled when selecting '
        'disable_language_model but can also be selected '
        'independently of that flag using this one.'
    ),
    default=False,
    dest='exclude_from_elo_calculation',
)
# Parse command line arguments
args = parser.parse_args()

exclude_from_elo_calculation = args.exclude_from_elo_calculation
if args.disable_language_model:
  exclude_from_elo_calculation = True

# Append name of agent to list of agents that already ran and throw error if it
# is already present.
if not exclude_from_elo_calculation:
  agents_list_path = f'agents__{args.model_name}__{args.embedder_name}.txt'
  agents_list = file_utils.read_csv(
      file_path=agents_list_path, unpack_singleton_rows=True)

  if args.agent_name in agents_list:
    raise ValueError('f{args.agent_name} was already evaluated.')
  else:
    file_handle = open(agents_list_path, 'a', encoding='utf-8')
    file_handle.write(f'{args.agent_name}\n')
    file_handle.close()

# Load the agent config with importlib
IMPORT_AGENT_BASE_DIR = 'concordia.factory.agent'
agent_module = importlib.import_module(
    f'{IMPORT_AGENT_BASE_DIR}.{args.agent_name}'
)

st_model = sentence_transformers.SentenceTransformer(
    f'sentence-transformers/{args.embedder_name}'
)

evaluation_results = {}
# Note: we could parallelize this loop.
for scenario_name, scenario_config in scenarios_lib.SCENARIO_CONFIGS.items():
  print(f'Running scenario: {scenario_name}')
  # Language Model setup
  model = utils.language_model_setup(args)
  # Setup sentence encoder
  embedder = lambda x: st_model.encode(x, show_progress_bar=False)

  # Run several simulations per scenario
  simulation_outcomes = []
  focal_per_capita_scores_to_average = []
  background_per_capita_scores_to_average = []
  ungrouped_per_capita_scores_to_average = []
  for repetition_idx in range(args.num_repetitions_per_scenario):
    measurements = measurements_lib.Measurements()
    runnable_simulation = scenarios_lib.build_simulation(
        scenario_config=scenario_config,
        model=model,
        focal_agent_module=agent_module,
        embedder=embedder,
        measurements=measurements,
    )
    # Run the simulation
    outcome, text_results_log = runnable_simulation()
    simulation_outcomes.append(outcome)
    if scenario_config.focal_is_resident:
      focal_scores = list(outcome.resident_scores.values())
      background_scores = list(outcome.visitor_scores.values())
    else:
      focal_scores = list(outcome.visitor_scores.values())
      background_scores = list(outcome.resident_scores.values())
    # Ungrouped scores do not differentiate between focal and background.
    ungrouped_scores = focal_scores + background_scores
    # Calculate per capita scores.
    focal_per_capita_score = np.mean(focal_scores)
    focal_per_capita_scores_to_average.append(focal_per_capita_score)
    print(f'  Focal per capita score: {focal_per_capita_score}')
    background_per_capita_score = np.mean(background_scores)
    background_per_capita_scores_to_average.append(background_per_capita_score)
    print(f'  Background per capita score: {background_per_capita_score}')
    ungrouped_per_capita_score = np.mean(ungrouped_scores)
    ungrouped_per_capita_scores_to_average.append(ungrouped_per_capita_score)
    print(f'  Ungrouped per capita score: {ungrouped_per_capita_score}')
    # Write the full text log as an HTML file in the current working directory.
    html_filename = (
        f'{scenario_name}_'
        + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        + '.html'
    )
    file_handle = open(html_filename, 'a', encoding='utf-8')
    file_handle.write(text_results_log)
    file_handle.close()
  # Average scores over repetitions and save results for all repetitions in a
  # json-serializable format.
  evaluation_results[scenario_name] = logging_lib.ScenarioResult(
      scenario=scenario_name,
      focal_agent=args.agent_name,
      background_agent=scenario_config.background_agent_module,
      focal_per_capita_score=np.mean(focal_per_capita_scores_to_average),
      background_per_capita_score=np.mean(
          background_per_capita_scores_to_average
      ),
      ungrouped_per_capita_score=np.mean(
          ungrouped_per_capita_scores_to_average
      ),
      simulation_outcomes=tuple(simulation_outcomes),
      focal_is_resident=scenario_config.focal_is_resident,
      api_type=args.api_type,
      model=args.model_name,
      embedder=args.embedder_name,
      disable_language_model=args.disable_language_model,
      exclude_from_elo_calculation=args.exclude_from_elo_calculation,
  )

# Save evaluation results for all scenarios with this agent to one json file.
json_filename = (
    f'{args.agent_name}__{args.model_name}__{args.embedder_name}.json'
)
idx = 0
with open(json_filename, 'a', encoding='utf-8') as file_handle:
  file_handle.write('[\n')
  for scenario_name, scenario_result in evaluation_results.items():
    json_str = evaluation_results[scenario_name].to_json()
    if idx < len(scenarios_lib.SCENARIO_CONFIGS) - 1:
      json_str += ',\n'
    file_handle.write(json_str)
    idx += 1
  file_handle.write('\n]')
