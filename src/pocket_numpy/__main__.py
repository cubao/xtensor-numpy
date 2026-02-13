import sys
import subprocess
from pathlib import Path


if __name__ == '__main__':
    exe = f'{Path(__file__).parent}/pocketpy.exe'
    args = sys.argv[1:]
    sys.exit(subprocess.call([exe, *args]))
