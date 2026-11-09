import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def run(data='data/archive', output='reports/pipeline', user_id=None):
    root = Path(__file__).resolve().parents[1]
    data = Path(data).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(root / 'Notebooks/Hybrid.ipynb', as_version=4)
    notebook.cells = [cell for cell in notebook.cells
                      if 'original_results' not in cell.metadata.get('tags', [])]
    env = dict(os.environ, BOOKMATCH_DATA=str(data), BOOKMATCH_OUTPUT=str(output),
               BOOKMATCH_USER='' if user_id is None else str(user_id))

    def progress(cell, cell_index):
        if cell.cell_type == 'markdown':
            for line in cell.source.splitlines():
                if line.startswith('## '):
                    print(line[3:], flush=True)

    client = NotebookClient(notebook, timeout=1800, kernel_name='python3',
                            resources={'metadata': {'path': str(root)}}, on_cell_start=progress)
    client.execute(env=env)
    for cell in notebook.cells:
        for result in cell.get('outputs', []):
            if 'data' in result:
                result['data'].pop('text/html', None)
    nbformat.write(notebook, output / 'Hybrid.ipynb')
    print(f'Результаты: {output}', flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Обучение и оценка BookMatch')
    parser.add_argument('--data', default='data/archive')
    parser.add_argument('--output', default='reports/pipeline')
    parser.add_argument('--user-id', type=int)
    args = parser.parse_args()
    run(args.data, args.output, args.user_id)
