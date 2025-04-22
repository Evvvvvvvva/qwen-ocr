from convert2sharegpt import convert_latex_formulas, convert_hmer
from extract_gt import extract_gt
from preprocess_gt import preprocess_gt
from merge_gt import merge_gt

# make sure to download the dataset first and place it in ./data
print("Converting latex formulas...")
convert_latex_formulas()
print("Converting hmer dataset...")
convert_hmer()
print("Running data filtering and tokenization for printed MER...")
# Get ground truth latex code
extract_gt()
# Tokenize ground truth latex code and drop broken entries
preprocess_gt()
# Update parquets
merge_gt()

