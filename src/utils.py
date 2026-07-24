import sys
import pandas as pd
import numpy as np
import pytest
import requests
import tqdm

def verify_environment():
    print("Environment Verification report: ")

print("Python version: ", sys.version)
print("Pandas version: ", pd.__version__)
print("NumPy version: ", np.__version__)
print("Pytest version: ", pytest.__version__)
print("Requests version: ", requests.__version__)
print("Tqdm version: ", tqdm.__version__)

print("Environment verification completed successfully.")

if __name__ == "__main__":
    verify_environment()