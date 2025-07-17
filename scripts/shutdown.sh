#!/bin/bash
# Kill, ignore errors
kill -9 $(cat pids.txt | awk '{print $2}')
