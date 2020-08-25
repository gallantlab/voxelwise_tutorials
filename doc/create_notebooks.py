if __name__ == "__main__":
    import os
    import shutil

    # check if `archive_name` exists
    archive_name = os.path.join('_auto_examples', '_auto_examples_jupyter.zip')
    if not os.path.exists(archive_name):
        raise RuntimeError(
            f"{archive_name} does not exist, please run `make html` first.")

    # Unpack `archive_name`
    extract_dir = os.path.join('..', 'tutorials', 'notebooks')
    shutil.unpack_archive(archive_name, extract_dir=extract_dir)
    print(f'Extracted {archive_name} to {extract_dir}')

    # copy the README.rst files
    tutorial_dir = os.path.join('..', 'tutorials')
    for file_or_dir in os.listdir(tutorial_dir):
        if os.path.isdir(os.path.join(tutorial_dir, file_or_dir)):
            if file_or_dir == "notebooks":
                continue
            source = os.path.join(tutorial_dir, file_or_dir, 'README.rst')
            destination = os.path.join(tutorial_dir, 'notebooks', file_or_dir,
                                       'README.rst')
            shutil.copyfile(source, destination)
            print(f'Copied {source} to {destination}')
