# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 11:14:36 2025

@author: mcallahan
"""

import os

# -------------------------------------------------
# USER SETTINGS
# -------------------------------------------------
folder = r"C:\Users\mcallahan\Untraditionally Classic\FlyGuys\DecemberSurvey\Dember16\WS26Arts"   # <-- change this
old_str = "09"
new_str = "16"
import os


# -------------------------------------------------
# RENAME FILES BY INDEX
# -------------------------------------------------
for fname in os.listdir(folder):
    old_path = os.path.join(folder, fname)

    # skip folders
    if not os.path.isfile(old_path):
        continue

    # safety: filename must be long enough
    if len(fname) < 10:
        continue

    # replace characters at indices 8–9 with "16"
    new_name = fname[:10] + "16" + fname[12:]
    new_path = os.path.join(folder, new_name)

    if new_name != fname:
        print(f"Renaming: {fname} → {new_name}")
        os.rename(old_path, new_path)
