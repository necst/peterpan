import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, "build")
sys.path.append(build_dir)

import peterpan_cpp

args = [
    "/home/gsorrentino/CT/",
    "/home/gsorrentino/PET/",
    "32",
    "--type=png",
    "--mask=otsu",
    "--gradient=itk",
    "--num_levels=4",
    "--num_iter=5"
]

print("Launching PeterPan from Python...")
peterpan_cpp.run(args)
print("Execution finished.")