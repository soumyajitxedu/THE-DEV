import os
import pkg_resources
from tqdm import tqdm

total_size = 0

# Get the list of packages, but handle it if the system returns None
working_set = pkg_resources.working_set
packages = list(working_set) if working_set is not None else []

print(f"Scanning {len(packages)} packages. This might take a few minutes...")

# tqdm adds a progress bar [====>] and automatically calculates an ETA
for dist in tqdm(packages, desc="Calculating size", unit="pkg"):
    package_path = dist.location
    if os.path.exists(package_path): # type: ignore
        # Recursively walk through every folder inside the package
        for dirpath, dirnames, filenames in os.walk(package_path): # type: ignore
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total_size += os.path.getsize(fp)
                except OSError:
                    pass # Skip files that might be locked or inaccessible

size_mb = total_size / (1024 ** 2)
size_gb = total_size / (1024 ** 3)

print(f"\n✅ Total storage used by Python libraries: {size_mb:.2f} MB ({size_gb:.2f} GB)")