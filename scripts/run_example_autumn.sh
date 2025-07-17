#! /bin/bash

MODEL="google/gemini-2.5-flash-preview"
LLM_PROVIDER="openrouter"
AGENT="autumn_llm_interactive_agent_v1"
echo "Running example with $MODEL, $LLM_PROVIDER, $AGENT"

mkdir -p logs/$MODEL

python -m python_examples.autumnbench.llm_agent --logfile logs/$MODEL/log_agent.log > logs/$MODEL/log_agent.txt 2>&1 &
pid2=$!

python -m python_examples.autumnbench.evaluation_controller > logs/$MODEL/log_evaluation_controller.txt 2>&1 &
pid3=$!

python -m python_examples.autumnbench.run --output-csv logs/$MODEL/results.csv --model $MODEL --llm-provider $LLM_PROVIDER --agent $AGENT > logs/$MODEL/log_run_evaluation.txt 2>&1 &
pid4=$!

echo "PIDs: $pid2 $pid3 $pid4" > logs/$MODEL/pids.txt

# Kill:
# kill -9 $(cat pids.txt | awk '{print $2}')
