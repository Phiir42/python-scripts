# run_largearea_regex.py
import os
import re
import subprocess, sys

# ---- USER SETTINGS ---------------------------------------------------------
DATA_ROOT = r"C:\Users\clchr\OneDrive - Stanford\Research Documents\AD Project\2025"
CONFIG_DIR = r"C:\Users\clchr\OneDrive - Stanford\Research Documents\Python Scripts\config_files"

# 1) Which dataset folders to process (recursive match)
FOLDER_PATTERN = re.compile(r"^AD[34][d-f]$", re.IGNORECASE)   # AD3d–f, AD4d–f

# 2) How to map folder name -> config file
# Option A (default): "config_<folder>.py"  e.g., AD3e -> config_AD3e.py
CONFIG_TEMPLATE = "config_{name}.py"

# Option B (overrides): explicit pattern -> config filename
# First matching rule wins; leave empty {} if you only want the template behavior.
CONFIG_RULES = {
    # Example overrides:
     # r"^AD3[de]$": "config_AD3d.py",
     # r"^AD4[f]$" : "config_AD4d.py",
}
# ---------------------------------------------------------------------------


def select_config_for_folder(folder_name: str) -> str | None:
    """
    Return an absolute path to the config for this folder, or None if not found.
    Matching priority:
      1) First rule in CONFIG_RULES whose regex matches folder_name.
      2) CONFIG_TEMPLATE with {name}=folder_name.
    """
    # 1) explicit regex rules
    for pat, cfg_file in CONFIG_RULES.items():
        if re.match(pat, folder_name, re.IGNORECASE):
            cfg_path = os.path.join(CONFIG_DIR, cfg_file)
            return cfg_path if os.path.isfile(cfg_path) else None

    # 2) template
    templated = CONFIG_TEMPLATE.format(name=folder_name)
    cfg_path = os.path.join(CONFIG_DIR, templated)
    return cfg_path if os.path.isfile(cfg_path) else None


def run_on_regex_sets(data_root: str) -> None:
    """
    Recursively find subfolders under data_root whose names match FOLDER_PATTERN,
    resolve a config for each, and invoke largearea_layers in a fresh subprocess.
    """
    found = False
    any_fail = False
    partials = []
    hardfails = []
    processed = []

    for dirpath, dirnames, _ in os.walk(data_root):
        base = os.path.basename(dirpath)
        if not FOLDER_PATTERN.fullmatch(base):
            continue
        found = True
        cfg = select_config_for_folder(base)
        if cfg is None:
            print(f"⚠️ Skipping {base}: no config found.")
            continue

        print(f"▶ Processing {base} in isolated subprocess")
        cmd = [sys.executable, "-m", "largearea_layers", cfg, dirpath]

        # Capture output so we can show clear status lines
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)

        # Echo the child stdout/stderr (so you keep the detailed per-ND2 logs)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            # stderr usually has tracebacks if any
            print(result.stderr, end="")

        rc = result.returncode
        if rc == 0:
            print(f"✓ Done: {base}")
            processed.append((base, "ok"))
        elif rc == 2:
            print(f"❗ Partial failure in {base} (some ND2s failed)")
            any_fail = True
            partials.append(base)
            processed.append((base, "partial"))
        else:
            print(f"❌ {base} failed (exit code {rc})")
            any_fail = True
            hardfails.append(base)
            processed.append((base, "fail"))

    if not found:
        print(f"No matching folders under: {data_root}")
        return

    # Final summary
    print("\n=== Summary ===")
    for base, status in processed:
        tag = {"ok":"✓","partial":"❗","fail":"❌"}[status]
        print(f"{tag} {base}: {status}")
    if partials:
        print(f"\nFolders with partial failures: {', '.join(partials)}")
    if hardfails:
        print(f"Folders that failed: {', '.join(hardfails)}")

    # Optional non-zero exit for CI / batch awareness
    if any_fail:
        # Propagate a non-zero exit to parent shell
        sys.exit(2)


if __name__ == "__main__":
    run_on_regex_sets(DATA_ROOT)
