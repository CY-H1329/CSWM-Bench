#!/usr/bin/env python3
"""
Patch SpatialRGPT vision_encoder.py for Python 3.9 compatibility.

The SpatialRGPT repo uses `match` (Python 3.10+). This script replaces
it with if/elif for Python 3.9.

Usage on H100 (before running eval):
  export SPATIALRGPT_PATH=/path/to/SpatialRGPT
  python scripts/stvqa7k/patch_spatialrgpt_py39.py
"""
import os
import re
import sys


def main():
    path = os.environ.get("SPATIALRGPT_PATH")
    if not path or not os.path.isdir(path):
        print("SPATIALRGPT_PATH not set or invalid. Skipping patch.")
        return 0

    fp = os.path.join(path, "llava", "model", "multimodal_encoder", "vision_encoder.py")
    if not os.path.isfile(fp):
        print(f"File not found: {fp}")
        return 1

    with open(fp, "r") as f:
        content = f.read()

    if "match interpolate_mode:" not in content:
        print("Patch already applied or file structure changed. Skipping.")
        return 0

    # Replace match/case with if/elif. The "linear" block content is indented
    # 4 spaces under case; under if it needs 4 fewer spaces.
    pattern = re.compile(
        r'        match interpolate_mode:\n'
        r'            case "linear":\n'
        r'((?:^[ ]{16,}.*\n)+)'
        r'            case _:\n'
        r'                raise NotImplementedError',
        re.MULTILINE,
    )

    def replacer(m):
        linear_block = m.group(1).rstrip("\n")
        # Reduce indentation by 4 spaces for each line
        new_lines = [
            line[4:] if len(line) >= 4 and line.startswith("    ") else line
            for line in linear_block.split("\n")
        ]
        new_block = "\n".join(new_lines)
        return (
            '        if interpolate_mode == "linear":\n'
            f'{new_block}\n'
            '        else:\n'
            '            raise NotImplementedError'
        )

    content, n = pattern.subn(replacer, content, count=1)
    if n == 0:
        print("Could not find expected match block. File may have changed.")
        return 1

    with open(fp, "w") as f:
        f.write(content)

    print(f"Patched {fp} for Python 3.9 compatibility.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
