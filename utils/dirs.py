import os


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def clear_dirs(which_folders=('loaders', 'logs', 'checkpoints'), name='run1'):
    for folder in which_folders:
        folder_path = os.path.join(folder, name)
        ensure_dir(folder_path)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
