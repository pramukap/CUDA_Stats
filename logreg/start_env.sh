#!/bin/bash

virtualenv env
source env/bin/activate
pip install -r requirements.txt
echo ""
echo ""
echo "Press CTRL+D to exit this Python env"
bash
