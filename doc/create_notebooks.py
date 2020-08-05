
if __name__ == "__main__":
    import os
    import shutil

    archive_name = "auto_examples/auto_examples_jupyter.zip"
    if not os.path.exists(archive_name):
        raise RuntimeError(
            f"{archive_name} does not exist, please run `make html` first.")

    extract_dir = "../tutorials/notebooks/"
    shutil.unpack_archive(archive_name, extract_dir=extract_dir)
    print(f'Extracted {archive_name} to {extract_dir}')
