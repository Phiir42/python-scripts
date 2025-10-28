# run_largearea_regex.py
import os
import re
import subprocess, sys

# Ensure subprocesses can import largearea_layers and lipid_analysis
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
SCRIPTS_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

# Prefer importing to discover the exact file location; fall back to join if import fails.
try:
    # Ensure the parent “Python Scripts” is importable in this parent process
    if SCRIPTS_ROOT not in sys.path:
        sys.path.insert(0, SCRIPTS_ROOT)
    import largearea_layers as _lal
    LAL_SCRIPT = os.path.abspath(_lal.__file__)
except Exception:
    LAL_SCRIPT = os.path.join(SCRIPTS_ROOT, "largearea_layers.py")

# ---- USER SETTINGS ---------------------------------------------------------
DATA_ROOT = r"D:\OneDrive - Stanford\Research Documents\AD Project\2025"
CONFIG_DIR = r"D:\OneDrive - Stanford\Research Documents\Python Scripts\config_files"

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
        print(f"  ↳ cfg={cfg}")

        # Run the script by absolute path (robust even under Spyder/Anaconda)
        cmd = [sys.executable, LAL_SCRIPT, cfg, dirpath]
        
        lal_dir = os.path.dirname(LAL_SCRIPT)
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=lal_dir  # execute from the folder that actually contains largearea_layers.py
        )

        # Echo the child stdout/stderr (so you keep the detailed per-ND2 logs)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            # stderr usually has tracebacks if any
            print(result.stderr, end="")

        rc = result.returncode
        # Prefer reading the child’s _STATUS.txt if present
        status_txt = os.path.join(dirpath, "LargeArea", "_STATUS.txt")
        if os.path.isfile(status_txt):
            try:
                with open(status_txt, "r", encoding="utf-8") as fh:
                    txt = fh.read()
                # Parse counts from the status file
                # Expected line: "Succeeded: X | Failed: Y | Total: Z"
                import re as _re
                m = _re.search(r"Succeeded:\s*(\d+)\s*\|\s*Failed:\s*(\d+)\s*\|\s*Total:\s*(\d+)", txt)
                if m:
                    ok = int(m.group(1)); fail = int(m.group(2))
                    if ok > 0 and fail == 0:
                        print(f"✓ Done: {base}")
                        processed.append((base, "ok"))
                    elif ok > 0 and fail > 0:
                        print(f"❗ Partial failure in {base} (some ND2s failed)")
                        any_fail = True
                        partials.append(base)
                        processed.append((base, "partial"))
                    else:
                        print(f"❌ {base} failed (no ND2s succeeded)")
                        any_fail = True
                        hardfails.append(base)
                        processed.append((base, "fail"))
                else:
                    # Couldn’t parse: fall back to rc
                    if rc == 0:
                        print(f"✓ Done: {base}")
                        processed.append((base, "ok"))
                    else:
                        print(f"❌ {base} failed (exit code {rc})")
                        any_fail = True
                        hardfails.append(base)
                        processed.append((base, "fail"))
            except Exception:
                # Read/parsing failed — fall back to rc
                if rc == 0:
                    print(f"✓ Done: {base}")
                    processed.append((base, "ok"))
                else:
                    print(f"❌ {base} failed (exit code {rc})")
                    any_fail = True
                    hardfails.append(base)
                    processed.append((base, "fail"))
        else:
            # No status file — this is a hard fail regardless of rc (script likely didn’t run)
            print(f"❌ {base} failed before status was written (exit code {rc})")
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
