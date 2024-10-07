import os
import shutil
import argparse
import re
import imageio.v3 as iio

def convert_dds_to_png(dds_file_path, output_path):
    try:
        img = iio.imread(dds_file_path)
        iio.imwrite(output_path, img)
        print(f"Converted '{dds_file_path}' to '{output_path}'")
    except Exception as e:
        print(f"Error converting '{dds_file_path}': {e}")

def process_folder(folder_path):
    delete_folder = os.path.join(folder_path, "delete")
    if not os.path.exists(delete_folder):
        os.makedirs(delete_folder)

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)  # Full path for the current file
        if os.path.isfile(file_path):
            if file.lower().endswith('.dds'):
                png_file_path = os.path.splitext(file_path)[0] + '.png'
                convert_dds_to_png(file_path, png_file_path)
                shutil.move(file_path, os.path.join(delete_folder, file))

            elif file.lower().endswith('.gltf'):
                try:
                    with open(file_path, 'r') as gltf_file:
                        content = gltf_file.read()

                    updated_content = re.sub(r'\.dds', '.png', content, flags=re.IGNORECASE)

                    with open(file_path, 'w') as gltf_file:
                        gltf_file.write(updated_content)

                    print(f"Updated '{file_path}' to replace all '.dds' with '.png'")

                except Exception as e:
                    print(f"Error processing '{file_path}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .dds files to .png and update .gltf references.")
    parser.add_argument("folder", nargs='?', default=os.getcwd(), help="Folder to process (default: current folder)")
    args = parser.parse_args()
    process_folder(args.folder)