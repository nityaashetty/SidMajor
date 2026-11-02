import pyedflib
import numpy as np
import os

def load_edf(filepath):
    f = pyedflib.EdfReader(filepath)
    n_channels = f.signals_in_file
    signal_labels = f.getSignalLabels()
    n_samples = f.getNSamples()[0]
    
    signals = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        signals[i, :] = f.readSignal(i)
    
    f.close()
    return signals, signal_labels

if __name__ == "__main__":
    edf_folder = r"D:\Major\caueeg-dataset\signal\edf"
    first_file = os.listdir(edf_folder)[0]
    filepath = os.path.join(edf_folder, first_file)
    
    print(f"Loading: {first_file}")
    signals, labels = load_edf(filepath)
    
    print(f"Signal shape: {signals.shape}")
    print(f"Channels: {labels}")
    print(f"Sample rate check - first channel samples: {signals.shape[1]}")