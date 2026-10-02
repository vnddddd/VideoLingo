import os
import shutil
from core.utils.output_names import output_filename

def delete_dubbing_files():
    files_to_delete = [
        os.path.join("output", "dub.wav"),
        os.path.join("output", "dub.mp3"),
        os.path.join("output", "dub_loudnorm.mp3"),
        os.path.join("output", "dub.srt"),
        os.path.join("output", "output_dub.mp4")
    ]
    for name in ("dub.mp3", "dub.srt"):
        exported = os.path.join("output", output_filename(name))
        if exported not in files_to_delete:
            files_to_delete.append(exported)
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
        else:
            print(f"File not found: {file_path}")
    
    segs_folder = os.path.join("output", "audio", "segs")
    if os.path.exists(segs_folder):
        try:
            shutil.rmtree(segs_folder)
            print(f"Deleted folder and contents: {segs_folder}")
        except Exception as e:
            print(f"Error deleting folder {segs_folder}: {str(e)}")
    else:
        print(f"Folder not found: {segs_folder}")

if __name__ == "__main__":
    delete_dubbing_files()
