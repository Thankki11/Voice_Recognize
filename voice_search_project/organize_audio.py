"""
Organize audio files into folders by sentence group.

Naming convention:
  arctic_a0001, arctic_a0001(1), arctic_a0001(3) ... arctic_a0001(7)
    → all go into folder: audio1

  arctic_a0002, arctic_a0002(1), ... arctic_a0002(7)
    → all go into folder: audio2
"""

import os
import re
import shutil
from pathlib import Path


def organize_audio_files(source_dir: str, dry_run: bool = False):
    """
    Scan source_dir for files matching arctic_aXXXX[(...)] pattern
    and move them into audioN subfolders.

    Args:
        source_dir: Directory containing the audio files.
        dry_run:    If True, only print what would happen without moving files.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"[ERROR] Directory not found: {source_dir}")
        return

    # Matches: arctic_a0001(3).wav  OR  arctic_a0001.wav  (extension optional)
    pattern = re.compile(r'^(arctic_a(\d+))(\(\d+\))?(\..+)?$', re.IGNORECASE)

    # Collect files grouped by sentence number
    groups: dict[int, list[Path]] = {}

    for file in source_path.iterdir():
        if not file.is_file():
            continue
        m = pattern.match(file.name)
        if not m:
            continue
        sentence_num = int(m.group(2))  # e.g. "0002" → 2
        groups.setdefault(sentence_num, []).append(file)

    if not groups:
        print("[INFO] No matching files found.")
        return

    print(f"[INFO] Found {len(groups)} sentence group(s).")

    for sentence_num in sorted(groups):
        folder_name = f"audio{sentence_num}"
        dest_folder = source_path / folder_name

        if not dry_run:
            dest_folder.mkdir(exist_ok=True)

        files = sorted(groups[sentence_num])
        print(f"\n  {folder_name}/  ({len(files)} file(s))")
        for f in files:
            dest = dest_folder / f.name
            print(f"    {f.name}")
            if not dry_run:
                shutil.move(str(f), str(dest))

    if dry_run:
        print("\n[DRY RUN] No files were moved. Remove dry_run=True to apply.")
    else:
        print("\n[DONE] All files organized successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Organize arctic_aXXXX audio files into audioN folders."
    )
    parser.add_argument(
        "source_dir",
        help="Path to the directory containing the audio files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without moving any files.",
    )
    args = parser.parse_args()

    organize_audio_files(args.source_dir, dry_run=args.dry_run)
