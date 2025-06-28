import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from erftools.preprocessing import Download_GFS_Data
from erftools.preprocessing import ReadGFS_3DData

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 WriteICFromGFSData.py <input_filename>")
        sys.exit(1)

input_filename = sys.argv[1]
filename, area = Download_GFS_Data(input_filename);
print("Filename is ", filename);

is_IC = True
print(f"Processing file: {filename}")
ReadGFS_3DData(filename, area, is_IC)
