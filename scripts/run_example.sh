#! /bin/bash
echo "Running example"

python -m python_examples.text_adventures.environment &
pid1=$!

python -m python_examples.text_adventures.agent &
pid2=$!

python -m python_examples.text_adventures.evaluation_controller &
pid3=$!

python -m python_examples.text_adventures.run

echo "PIDs: $pid1 $pid2 $pid3" > pids.txt
# Kill:
# kill -9 $(cat pids.txt | awk '{print $2}')

