import os
import glob
import re

def update_script(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Look for the deberta-v3-large block
    # e.g.:
    # 	deberta-v3-large)
    # 	parameters=" --num_train_epochs 2 \
    # 	...
    # 	--cls_drop_out 0.3 "
    # 		;;

    pattern = r'([ \t]+)deberta-v3-large\)\n(.+?);;'

    # Find the block
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"Skipping {filepath}, deberta-v3-large block not found.")
        return

    indent = match.group(1)
    block_content = match.group(2)

    # Transform the block content
    # 1. Add init=deberta-v3-large
    # 2. Add --use_recurrent args inside parameters

    # Find where parameters string ends
    param_end_match = re.search(r'("\s*\n)', block_content)
    if not param_end_match:
        # Maybe it ends with " without newline
        param_end_match = re.search(r'("\s*)$', block_content)

    if not param_end_match:
        print(f"Failed to parse parameters in {filepath}")
        return

    param_end_idx = param_end_match.start()
    
    recurrent_args = r" \ \n" + indent + r"--use_recurrent True \ \n" + indent + r"--recurrent_layer 13 \ \n" + indent + r"--ponder_penalty 1e-3"
    
    modified_block = block_content[:param_end_idx] + recurrent_args + block_content[param_end_idx:]
    
    new_block = f"{indent}recurrent-deberta-v3-large)\n{indent}init=deberta-v3-large\n{modified_block};;"
    
    # Insert new block after the original block
    original_block = match.group(0)
    new_content = content.replace(original_block, original_block + "\n" + new_block)
    
    if new_content != content:
        with open(filepath, 'w', newline='\n') as f:
            f.write(new_content)
        print(f"Updated {filepath}")
    else:
        print(f"No changes made to {filepath}")

if __name__ == "__main__":
    base_dir = r"c:\Users\giuse\Downloads\RNN\DeBERTa\experiments\glue"
    scripts = glob.glob(os.path.join(base_dir, "*.sh"))
    for script in scripts:
        if script.endswith("download_data.sh"):
            continue
        update_script(script)
