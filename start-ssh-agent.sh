#!/bin/bash
# start this ssh-agent in the background

# For W11, you must kill all old ssh-agents first
# otherwise, you will have to enter the passphrase for your
# key constantly
pkill ssh-agent

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
