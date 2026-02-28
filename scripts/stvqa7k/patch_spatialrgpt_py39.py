#!/usr/bin/env python3
"""
Patch SpatialRGPT vision_encoder.py for Python 3.9 compatibility.

The SpatialRGPT repo uses `match` (Python 3.10+). This script replaces
it with if/elif for Python 3.9.

Usage on H100:
  export SPATIALRGPT_PATH=/path/to/SpatialRGPT
  python scripts/stvqa7k/patch_spatialrgpt_py39.py

  # Or with explicit path:
  python scripts/stvqa7k/patch_spatialrgpt_py39.py /path/to/SpatialRGPT
"""
import os
import re
import sys


def patch_file(fp: str) -> int:
    if not os.path.isfile(fp):
        print(f"File not found: {fp}")
        return 1

    with open(fp, "r") as f:
        content = f.read()

    if "match interpolate_mode:" not in content:
        print("Patch already applied or file structure changed. Skipping.")
        return 0

    # Flexible pattern: match any indentation (4 or 8 spaces typical)
    pattern = re.compile(
        r'^(\s*)match interpolate_mode:\s*\n'
        r'\1    case "linear":\s*\n'
        r'((?:\1        .*\n)+)'
        r'\1    case _:\s*\n'
        r'\1        raise NotImplementedError',
        re.MULTILINE,
    )

    def replacer(m):
        base_indent = m.group(1)
        linear_block = m.group(2).rstrip("\n")
        # Reduce indentation by 4 spaces for each line
        new_lines = []
        for line in linear_block.split("\n"):
            if len(line) >= 4 and line.startswith("    "):
                new_lines.append(line[4:])
            else:
                new_lines.append(line)
        new_block = "\n".join(new_lines)
        return (
            f'{base_indent}if interpolate_mode == "linear":\n'
            f'{new_block}\n'
            f'{base_indent}else:\n'
            f'{base_indent}    raise NotImplementedError'
        )

    content_new, n = pattern.subn(replacer, content, count=1)
    if n == 0:
        # Fallback: try 8-space base indent (method body)
        pattern2 = re.compile(
            r'^(        )match interpolate_mode:\n'
            r'\1    case "linear":\n'
            r'((?:\1        .*\n)+)'
            r'\1    case _:\n'
            r'\1        raise NotImplementedError',
            re.MULTILINE,
        )
        content_new, n = pattern2.subn(replacer, content, count=1)

    if n == 0:
        print("Could not find match block. File structure may have changed.")
        return 1

    with open(fp, "w") as f:
        f.write(content_new)

    print(f"Patched {fp} for Python 3.9 compatibility.")
    return 0


def main():
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        path = os.environ.get("SPATIALRGPT_PATH")

    if not path or not os.path.isdir(path):
        print("SPATIALRGPT_PATH not set or invalid. Usage: python patch_spatialrgpt_py39.py /path/to/SpatialRGPT")
        return 1

    fp = os.path.join(path, "llava", "model", "multimodal_encoder", "vision_encoder.py")
    return patch_file(fp)


if __name__ == "__main__":
    sys.exit(main())
