#!/bin/sh

# remove_cell_tags :: remove all cell

# remove input, keep output

#remove_input_tags

jupyter nbconvert "test_plots" --to pdf --TagRemovePreprocessor.remove_cell_tags='["hide_cell"]' --TagRemovePreprocessor.remove_input_tags='["hide_input"]' --no-prompt --output report
