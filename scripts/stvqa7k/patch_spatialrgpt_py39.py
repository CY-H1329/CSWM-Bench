#!/usr/bin/env python3
"""
Patch SpatialRGPT vision_encoder.py for Python 3.9 compatibility.

The SpatialRGPT repo uses `match` (Python 3.10+). This script replaces
it with if/elif for Python 3.9.

Usage on H100:
  export SPATIALRGPT_PATH=/path/to/SpatialRGPT
  python scripts/stvqa7k/patch_spatialrgpt_py39.py
"""
import os
import sys


def patch_file(fp: str) -> int:
    if not os.path.isfile(fp):
        print(f"File not found: {fp}", file=sys.stderr)
        return 1

    with open(fp, "r") as f:
        lines = f.readlines()

    if not any("match interpolate_mode:" in line for line in lines):
        return 0  # Already patched

    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Look for: "        match interpolate_mode:" (8 spaces) or similar
        stripped = line.lstrip()
        if stripped.startswith("match interpolate_mode:"):
            base_indent = line[: len(line) - len(stripped)]
            # Replace match with if
            new_lines.append(f'{base_indent}if interpolate_mode == "linear":\n')
            i += 1
            # Skip "case "linear":" line
            if i < len(lines) and 'case "linear"' in lines[i]:
                i += 1
            # Process content until "case _:"
            while i < len(lines):
                cl = lines[i]
                if "case _:" in cl:
                    new_lines.append(f"{base_indent}else:\n")
                    i += 1
                    if i < len(lines) and "raise NotImplementedError" in lines[i]:
                        err_line = lines[i]
                        err_indent = err_line[: len(err_line) - len(err_line.lstrip())]
                        # Reduce indent by 4 spaces
                        if len(err_indent) >= 4:
                            new_lines.append(" " * (len(err_indent) - 4) + err_line.lstrip())
                        else:
                            new_lines.append(err_line)
                        i += 1
                    break
                # Unindent content by 4 spaces
                if cl.startswith("    "):
                    new_lines.append(cl[4:])
                else:
                    new_lines.append(cl)
                i += 1
            continue
        new_lines.append(line)
        i += 1

    with open(fp, "w") as f:
        f.writelines(new_lines)

    return 0


def main():
    path = os.environ.get("SPATIALRGPT_PATH") or (sys.argv[1] if len(sys.argv) >= 2 else None)
    if not path or not os.path.isdir(path):
        return 1
    fp = os.path.join(path, "llava", "model", "multimodal_encoder", "vision_encoder.py")
    return patch_file(fp)


if __name__ == "__main__":
    sys.exit(main())
