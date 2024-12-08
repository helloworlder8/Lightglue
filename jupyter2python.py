

import json
import os

def ipynb_to_py(ipynb_file, output_py_file=None):
    """
    Convert a Jupyter Notebook (.ipynb) to a Python script (.py).
    Markdown cells are converted to comments, and code cells are preserved as is.

    Args:
        ipynb_file (str): Path to the input .ipynb file.
        output_py_file (str): Path to save the output .py file. If None, save with the same name as the input.
    """
    # Ensure the input file is valid
    if not ipynb_file.endswith('.ipynb'):
        raise ValueError("Input file must be a .ipynb file")

    # Determine the output .py file name
    if output_py_file is None:
        output_py_file = os.path.splitext(ipynb_file)[0] + ".py"

    # Read the .ipynb file
    with open(ipynb_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Prepare the output lines
    output_lines = []

    for cell in notebook.get('cells', []):
        if cell['cell_type'] == 'markdown':
            # Convert Markdown to comments
            markdown_lines = cell['source']
            # output_lines.append("# Markdown cell:")
            output_lines.extend(f"# {line.strip()}" for line in markdown_lines)
            output_lines.append("\n")
        elif cell['cell_type'] == 'code':
            # Preserve code as is
            code_lines = cell['source']
            # output_lines.append("# Code cell:")
            output_lines.extend(line for line in code_lines)
            output_lines.append("\n")
        else:
            # Ignore other cell types
            output_lines.append(f"# Skipped a {cell['cell_type']} cell\n")

    # Write to the output .py file
    with open(output_py_file, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)

    print(f"Conversion completed: {output_py_file}")


# Example usage
if __name__ == "__main__":
    # Replace 'example.ipynb' with your .ipynb file path
    ipynb_file_path = "demo.ipynb"
    ipynb_to_py(ipynb_file_path)
