# ECG Classification with ECG-Net and ECG-TCN 

This project uses the **ECG5000 dataset** to classify ECG signals into healthy and unhealthy categories using **ECG-Net**. If any unhealthy data is detected, the second model, **ECG-TCN**, is used to further classify the type of arrhythmia. The models were tested on a system with the following specifications:

- **GPU**: RTX 2060 Super
- **RAM**: 32GB
- **CPU**: Ryzen 5 3600X

## How to Use

1. Run `main.py`. 
2. You will be asked two inputs:
   - **Disable randomness? (y/n)**: Selecting `y` sets the random seed to 42, ensuring repeatable results, while selecting `n` will keep the seed random.
   - **Choose one of the following options**:
     ```
     1 - Test ECG-NET
     2 - Test ECG-TCN
     3 - Test ECG-NET + ECG-TCN
     4 - Train ECG-Net
     5 - Train ECG-TCN
     ```

### Options Explanation:
- **Option 1**: Tests the ECG-NET model with the testing data.
- **Option 2**: Tests the ECG-TCN model with the testing data.
- **Option 3**: Runs the ECG-NET model first to classify healthy and unhealthy data, then uses ECG-TCN to further classify the unhealthy data into one of the arrhythmia categories.
- **Option 4**: Trains the ECG-NET model, which classifies healthy heart ECG data from unhealthy ECG data.
- **Option 5**: Trains the ECG-TCN model, which specifically classifies the type of arrhythmia: R-T, PVC, SB/EB, or FVN.

### Citations:
- **ECG-NET**: Roy, M. et al. (2023) ‘ECG-net: A deep LSTM autoencoder for detecting anomalous ECG’, Engineering Applications of Artificial Intelligence, 124, p. 106484. doi:10.1016/j.engappai.2023.106484. 
- **ECG-TCN**: Ingolfsson, T.M. et al. (2021) ‘ECG-TCN: Wearable cardiac arrhythmia detection with a temporal convolutional network’, 2021 IEEE 3rd International Conference on Artificial Intelligence Circuits and Systems (AICAS) [Preprint]. doi:10.1109/aicas51828.2021.9458520. 

