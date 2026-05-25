#!/bin/bash
cd /home/user/dev/AI_GAME_ENGINE/flowforge
bunx tsc --noEmit -p packages/core/tsconfig.json > /home/user/dev/AI_GAME_ENGINE/flowforge/typecheck_output.txt 2>&1
echo "EXIT: $?" >> /home/user/dev/AI_GAME_ENGINE/flowforge/typecheck_output.txt
