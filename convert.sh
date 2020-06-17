#!/bin/sh

# USAGE: ./convert.sh notebook_name format output_name
# Note that the format can be pdf, latex or whatever pandoc accepts.

jupyter nbconvert $1 --to $2 --TagRemovePreprocessor.remove_cell_tags='["hide_cell"]' --TagRemovePreprocessor.remove_input_tags='["hide_input"]' --no-prompt --output $3
