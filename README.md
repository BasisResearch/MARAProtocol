# MARAProtocol
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is used as blueprint to implement MARAProtocol. The protocol is designed to be a language-agnostic interface to implement MARA environments and agents.
**MARAProtocol** is an open, language-agnostic interface built on Protocol Buffers that standardizes how interactive environments and learning agents exchange observations, actions, and rewards, no matter which programming language or domain they use. To do this, MARA protocols are currently defined in Protobuf.
By separating runtime orchestration from domain logic and shipping ready-made stubs, evaluation tooling, and a growing benchmark suite, it lets environment builders plug in new tasks quickly while enabling agent developers to test and compare algorithms across heterogeneous, high-bandwidth domains—from robotics to Autumn video-game worlds.

The protocols are stored in `protocols/*.proto` files. 

## Table of Contents
- Installation
- Examples
- License

## Installation
### Requirements
- Python >= 3.12 (see note on 3.13 pre-release)  
- `protoc` >= 3.0  
- macOS/Linux (Windows instructions TBD)

To install these dependencies and generate the corresponding Python stubs and library for implementing environments and agents, please execute the following scripts:

```bash
git clone https://github.com/BasisResearch/MARAProtocol.git
cd MARAProtocol
sh setup_script_mac.sh           # installs deps
sh scripts/generate_python.sh     # generates gRPC stubs into ./generated
```

### Obtaining the interpreter
We have a prebuilt interpreter for Python 3.13 on MacOSX. More prebuilts are available at [Autumn.cpp/Autumn.wasm repository](https://github.com/BasisResearch/Autumn.cpp/releases). For any different versions, please follow the build guide of the repository.

### Setting up environment variables
Following this, please create an `.env` file following the `.env_sample`. This file is mostly used for providing credentials for LLM Agents.
Our supported LLM Agents includes (and will grow if needed): Ollama, OpenAI, Claude, MLX, Gemini. We also support framework OpenRouter framework for easily switch between different LLM providers.

You can also setup the API key directly, for example:
```bash
export OPENROUTER_API_KEY="YOUR_API_KEY_HERE"
```

We list the corresponding providers belows for quick start.

| **Provider**       | **Docs / Quick-start**                                                                         |
| ------------------ | ---------------------------------------------------------------------------------------------- |
| Ollama             | [REST API reference](https://ollama.readthedocs.io/en/api/)                                    |
| OpenAI             | [API key setup](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety) |
| Claude (Anthropic) | [Anthropic Developer Docs](https://docs.anthropic.com/)                                        |
| Apple MLX          | [MLX GitHub](https://github.com/ml-explore/mlx)                                                |
| Google Gemini      | [Gemini API docs](https://ai.google.dev/gemini-api/docs)                                       |
| OpenRouter         | [OpenRouter Quickstart](https://openrouter.ai/docs/quickstart)                                 |

Drop these variables into your local `.env`; the MARAProtocol tooling will load them automatically at runtime.

## Examples
We provide several examples:
1. A standard environment and random agent implementing MARAProtocol as a text adventure in `python_examples/text_adventures`.
2. [AutumnBench environments](./python_examples/autumnbench/): Masked Frame Prediction (MFP), Change Detection (or Change Detection) and Planning. Along with three types of agents: Random agent, LLM-based agent, and an agent that's built with [AutumnSynth](https://www.basis.ai/blog/autumn/) in mind.

### Downloading the Dataset
To run the AutumnBench examples, you first need to download the dataset. The dataset contains the tasks, programs, and prompts needed for the benchmark.

To download the public dataset, run the following script from the root of the repository:

```bash
./scripts/download_dataset.sh
```

This script requires the following tools to be installed:
- **curl**: The script uses `curl` to download files via HTTPS. It is pre-installed on most modern operating systems (like macOS and popular Linux distributions).
- **jq**: A lightweight and flexible command-line JSON processor. You can install it with `brew install jq` on macOS or find other installation methods [here](https://stedolan.github.io/jq/download/).

After the script finishes, the dataset will be available in the `python_examples/autumnbench/example_benchmark/` directory.

### AutumnBench Baselines
We also provide an example benchmark for Autumn that consists of a single program, this is stored in [Example Benchmark](./python_examples/autumnbench/example_benchmark/).

Once this is done, you can run the agent with either one of the following agents. Note that, protobuf codes are originally meant for creating gRPC for a language-agnostic interface. However, for simplicity, we usethem locally.


### Running Agents
```bash
python -m python_examples.autumnbench.run_no_server +experiment=debug data_dir=$(pwd)/python_examples/autumnbench/example_benchmark
```

More configurable parameters can be found in `python_examples/autumnbench/conf/config.yaml`. You can either specify a new experiment config in `conf/experiments/` or specify them at runtime. For example, to change the render mode you can simply run the following command.

```bash
python -m python_examples.autumnbench.run_no_server +experiment=debug data_dir=$(pwd)/python_examples/autumnbench/example_benchmark render_mode=image
```

If you would like to run with another model (say Claude 4 Opus) on OpenRouter, you can do the following:

```bash
python -m python_examples.autumnbench.run_no_server +experiment=debug data_dir=/python_examples/autumnbench/example_benchmark="anthropic/claude-opus-4"
```

You can also configure the environments you want to run on by changing the list in the `envs` parameter.  For the `task_name` parameter the options supported as `(mfp, cd, planning)`.

The main agent is the `UnifiedReactAgent` defined in [`llm_agent.py`](./python_examples/autumnbench/llm_agent.py), with some of the prompts defined in [`prompts.py`](./python_examples/autumnbench/prompts.py). The task type themselves are defined in [`concrete_envs.py`](./python_examples/autumnbench/concrete_envs.py). Adding a new environment should be done by adding it to the `AutumnBenchmark` repo directly.

We currently provide the following three agents:
- "autumn_llm_unified_interactive_agent_v1" # LLM-based agent
- "autumn_random_interactive_agent_v1"      # Random agent
- "autumn_simple_wm_agent"                  # Oracle autumnSynth agent

You can select the default desired agent in [`config.yaml`](`python_examples/autumnbench/conf/config.yaml`).

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Contributors
Dat Nguyen, Moksh Jain, Yichao Liang, Archana Warrier, Michelangelo Naim, Cambridge Yang, Zenna Tavares

**Open source contributors**: @hellohawaii
