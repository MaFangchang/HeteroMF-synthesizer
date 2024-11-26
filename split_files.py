import os
import random


def process_files(folder_path, prefix, output_file, sub_dataset):
    """
    folder_path (str): path of the folder.
    prefix (str): prefix of the file name.
    output_file (str): output file name.
    """
    file_names = []
    for pre in prefix:
        file_names.extend([os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.startswith(pre)])

    print(f"Number of {sub_dataset} files: {len(file_names)}")

    random.shuffle(file_names)

    with open(output_file, 'w') as f:
        for name in file_names:
            f.write(name + '\n')


if __name__ == '__main__':
    step_path = 'HeteroMF/steps'
    timestamp = {
        'train': ['20240622_111152'],
        'val': ['20240623_115138'],
        'test': ['20240623_164904']
    }

    for sub_data, stamp in timestamp.items():
        output_path = sub_data + '.txt'
        process_files(step_path, stamp, output_path, sub_data)
