import os,sys
import json
import argparse
if sys.version_info >= (3, 6):
    import zipfile
else:
    import zipfile36 as zipfile

def list_files_in_directory(directory_path):
    # List all files in the directory
    files = []
    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.wav'):   # only add .wav files
                files.append(os.path.join(root, filename))
    return files

def save_files_to_json(files, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(files, json_file, indent=4)

def make_json(directory_path, output_file):
    # Get the list of files and save to JSON
    files = list_files_in_directory(directory_path)
    save_files_to_json(files, output_file)


# create training set json
def extract_zip_dirs(data_dir= os.path.join(os.getcwd(),"data","dataset")):
    for zip_file in os.listdir(data_dir):
        if zip_file.endswith(".zip"):
            zip_path = os.path.join(data_dir, zip_file)
            extract_dir = os.path.join(data_dir, os.path.splitext(zip_file)[0])  # Remove .zip extension
            print(f"Extracting dir {zip_file}")    
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # Delete the original zip file
            os.remove(zip_path)
            print(f"Remove .zip dir {zip_file}") 
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_path', default='None')

    args = parser.parse_args()

    prefix = args.prefix_path if (args.prefix_path != 'None') else os.path.join(os.getcwd(),"data","dataset") #change to your adress
    
    #extract dataset dirs
    print("Extracting dataset directories from .zip")
    extract_zip_dirs()
    ## You can manualy modify the clean and noisy path
    # ----------------------------------------------------------#
    ## train_clean
    make_json(
        os.path.join(prefix, 'clean_trainset_28spk_wav/'),
        'data/train_clean.json'
    )

    ## train_noisy
    make_json(
        os.path.join(prefix, 'noisy_trainset_28spk_wav/'),
        'data/train_noisy.json'
    )
    # ----------------------------------------------------------#

    # ----------------------------------------------------------#
    # create valid set json
    ## valid_clean
    make_json(
         os.path.join(prefix, 'clean_testset_wav/'),
        'data/valid_clean.json'
    )

    ## valid_noisy
    make_json(
        os.path.join(prefix, 'noisy_testset_wav/'),
        'data/valid_noisy.json'
    )
    # ----------------------------------------------------------#

    # ----------------------------------------------------------#
    # create testing set json
    ## test_clean
    make_json(
       os.path.join(prefix, 'clean_testset_wav/'),
        'data/test_clean.json'
    )

    # ## test_noisy
    make_json(
       os.path.join(prefix, 'noisy_testset_wav/'),
        'data/test_noisy.json'
    )
    # ----------------------------------------------------------#


if __name__ == '__main__':
    main()
