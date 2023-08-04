#!/bin/bash
#cp models/*.model /mnt/e/transformers/.
#cp data/*.csv /mnt/e/transformers/.
(cd ~/PlayWithAI/transformers; tar -cvf - .)|(cd /mnt/e/transformers; tar -xvf -)
