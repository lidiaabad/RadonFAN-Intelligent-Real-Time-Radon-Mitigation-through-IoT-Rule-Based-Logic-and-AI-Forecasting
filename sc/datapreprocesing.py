import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import random
from scipy.interpolate import interp1d
import os
import scipy.ndimage
from numpy.lib.stride_tricks import sliding_window_view
'''
================================= MAIN DATA PREPROCESSING FUNCTIONS =============================
'''

class RadonDataset(Dataset):
    def __init__(self, X, y, yb, w_cl, w_reg, scaler=None, augment=False, train_std=None, segment_size=4):
        self.X = X
        self.y = y
        self.yb = yb
        self.w_cl = w_cl
        self.w_reg = w_reg
        self.scaler = scaler
        self.augment = augment
        self.train_std = train_std
        self.segment_size = segment_size
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        
        x = self.X[idx].copy().astype(np.float32)   # ensure numeric
        y = np.array(self.y[idx], dtype=np.float32) # forecast array or scalar
        yb = np.array(self.yb[idx], dtype=np.float32) # scalar 0/1
        w_cl = np.array(self.w_cl[idx], dtype=np.float32)
        w_reg = np.array(self.w_reg[idx], dtype=np.float32)
        #print(idx)
        
        if self.augment:
            #different augment probability to class 0 and 1 to ensure more for class=1
            if self.yb[idx]== 1: 
                prob=0.75
                factor=2.5
            else: 
                prob=0.25
                factor=1

            #jittering
            if random.random() < prob:
                noise_scale = factor* 0.05 * self.train_std  
                noise = np.random.normal(0, noise_scale, size=x.shape)
                x += noise
            #scaling
            if random.random() < prob:
                scale_factor = random.uniform(1-(0.1*factor), 1 + (0.1*factor))
                x *= scale_factor

            #time warping
            if random.random() < prob and len(x) > 3:
                orig_idx = np.arange(len(x))
                random_idx = np.linspace(0, len(x)-1, num=len(x))
                perturb = np.random.uniform(-3, 3, size=len(x))
                perturb_smooth = scipy.ndimage.gaussian_filter1d(perturb, sigma=(1*factor))  
                warped_idx = np.clip(np.arange(len(x)) + perturb_smooth, 0, len(x)-1)
                f = interp1d(orig_idx, x, axis=0, kind='cubic', fill_value="extrapolate")
                x = f(warped_idx)
        
            #magniture warping
            if random.random() <prob and len(x)> self.segment_size:
                n_segments = len(x) // self.segment_size
                for i in range(n_segments):
                    start = i * self.segment_size
                    end = start + self.segment_size
                    factor = random.uniform(1-(0.1*factor), 1 + (0.1*factor))  
                    x[start:end] *= factor
                    
            #masking
            if random.random() < 0.1*factor and len(x) > 4: 
                n = random.randint(1,len(x) //4)
                x[-n:] = 0.0 

        return (torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(yb, dtype=torch.float32), torch.tensor(w_cl, dtype=torch.float32), torch.tensor(w_reg, dtype=torch.float32))


def preprocess_data(file_path, lookback=216, forecast=6, threshold=300, numfeats=1, testonly=False, scaler_path=None):
    '''
    Function to preprocess the data
    Input: 
        - File path with csv 
    Output: 
        - X, y, y_bin, weights per instance groups for train, val and test
        - scaler on x and y
        - train standard deviation for augmentation 
    '''
    
    # --- Load, preprocess and clean ---
    data = pd.read_csv(file_path)
    drop_cols = ['entry_id', 'latitude', 'longitude', 'elevation', 'status']
    data = data.drop(drop_cols, axis=1)
    data.columns = ['DateTime', 'Radon', 'Ventilador', 'Umbral', 'Umbral alerta','Pendiente subida', 'Pendiente bajada', 'Peso Radon', 'Presion_indoor']
    data = data.drop(['Ventilador', 'Umbral', 'Umbral alerta', 'Pendiente subida', 'Pendiente bajada', 'Peso Radon', 'Presion_indoor'], axis=1)
    data['DateTime'] = pd.to_datetime(data['DateTime'], utc=True)
    data = data.sort_values(by='DateTime')
    start_date = pd.Timestamp("2024-01-10", tz="UTC")
    end_date   = pd.Timestamp("2024-03-18", tz="UTC")
    data_period = data[(data['DateTime'] >= start_date) & (data['DateTime'] <= end_date)]

    # --- Impute Rt (raw Radon) ---
    data_period.loc[data_period['Radon'] == 0.0, 'Radon'] = np.nan
    imputer = KNNImputer(n_neighbors=5)
    Rt = imputer.fit_transform(data_period[['Radon']]).flatten()

    # --- Compute derived features for ---
    n = len(Rt)
    if num_feats > 1: 
        cumsum_win = np.zeros(n)
        cumsum_win[:lookback+1] = np.sum(Rt[:lookback+1])  # inicial
        max_Rt_lookback = np.zeros(n)
        max_Rt_lookback[:lookback+1] = Rt[:lookback+1].max()
        count_above_thr = np.zeros(n, dtype=int)
        for t in range(lookback+1, n):
            cumsum_win[t] = np.sum(Rt[t - lookback + 1 : t + 1])
            max_Rt_lookback[t] = np.max(Rt[t - lookback + 1 : t + 1])
            window = Rt[t - lookback + 1 : t + 1]
            count_above_thr[t] = np.sum(window > (threshold * 0.75))

    if num_feats > 4: 
        Rv = np.zeros_like(Rt, dtype=float)
        Rv[0] = Rt[0]
        for t in range(1, n):
            Rv[t] = 0.5 * Rt[t] + 0.5 * Rv[t-1]
        Dt = Rt - Rv
        slope = np.zeros(n)
        slope[1:] = (Rt[1:] - Rt[:-1])
        slope3 = np.zeros(n)
        slope3[3:] = (Rt[3:] - Rt[:-3])
        slope6 = np.zeros(n)
        slope6[6:] = (Rt[6:] - Rt[:-6])
        slope12 = np.zeros(n)
        slope12[12:] = (Rt[12:] - Rt[:-12])
        sec_der=np.zeros(n)
        sec_der[1:] = (slope[1:] - slope[:-1])
        windows3 = sliding_window_view(Rt, window_shape=3)
        var_3 = np.var(windows3, axis=1, ddof=1)
        var3 = np.concatenate([[var_3[0]]*3, var_3[:-1]])
        windows6 = sliding_window_view(Rt, window_shape=6)
        var_6 = np.var(windows6, axis=1, ddof=1)
        var6 = np.concatenate([[var_6[0]]*6, var_6[:-1]])
        dist_to_threshold =  Rt - threshold
        sv = np.zeros_like(slope, dtype=float)4
        sv[0] = slope[0]
        for t in range(1, n):
            sv[t] = 0.5 * slope[t] + 0.5 * sv[t-1]
            
        lastrelRt= np.zeros(n)
        lastrelslope= np.zeros(n)
        lastrelRt[:lookback+1] = 0
        lastrelslope[:lookback+1] = 0
        for t in range((lookback+1), n):
            lastrelRt[t]= Rt[t] - np.mean(Rt[t - lookback + 1 : t + 1])
            lastrelslope[t] = slope[t] - np.mean(slope[t - lookback + 1 : t + 1])
        
        windows3 = sliding_window_view(Rt, window_shape=3)
        mean_3 = np.mean(windows3, axis=1)
        mean_3 = np.concatenate([[mean_3[0]]*3, mean_3[:-1]])
        windows12 = sliding_window_view(Rt, window_shape=12)
        mean_12 = np.mean(windows12, axis=1)
        mean_12 = np.concatenate([[mean_12[0]]*12, mean_12[:-1]])
        ma_ratio=mean_3 / (mean_12 + 1e-6)
        acel=slope3-slope

    # --- Combine features ---
    if numfeats==16: 
        feats = np.stack([Rt, Rv, Dt, slope, slope3, var3, slope6, var6, sv, lastrelRt, lastrelslope, cumsum_win, max_Rt_lookback, count_above_thr, ma_ratio, acel], axis=-1) 
    elif numfeats==4: 
        feats = np.stack([Rt, cumsum_win, max_Rt_lookback, count_above_thr], axis=-1)
    else: 
        feats = Rt.reshape(-1, 1)
    
    # ---- Binary Labels computation 
    y_bin_full = np.zeros(n, dtype=int)
    for i in range(forecast, n):
        if np.any(Rt[i-forecast+1:i+1] >= threshold):

    # --- Splits ---
    split_test = int(0.15 * n)
    split_val = int(0.30* n)

    # ---- Scaler: create or load and apply it to splits
    base = os.path.basename(file_path)
    name = os.path.splitext(base)[0]
    models_dir = "../models" if file_path.startswith("../data/") else "models"
    scaler_path_base = os.path.join(models_dir, f"{name}_scaler.pkl")

    if not testonly:
        # if we do not just want to test, but also to train
        # Split x and y
        train_feats = feats[split_val:]
        val_feats   = feats[split_test:split_val]
        test_feats  = feats[:split_test]
        y_bin_train = y_bin_full[split_val:]
        y_bin_val   = y_bin_full[split_test:split_val]
        y_bin_test  = y_bin_full[:split_test]
        # Fit scaler to x data, save it and then apply it to the three sets
        scaler = RobustScaler() 
        scaler.fit(train_feats)
        os.makedirs(models_dir, exist_ok=True)
        with open(scaler_path_base, "wb") as f:
            joblib.dump(scaler, f)
        train_scaled = scaler.transform(train_feats)
        val_scaled   = scaler.transform(val_feats)
        test_scaled  = scaler.transform(test_feats)
        # Fit y scaler using only Rt values and save it
        scaler_y = RobustScaler()
        scaler_y.fit(train_feats[:, [0]])
        with open(scaler_path_base.replace(".pkl", "_y.pkl"), "wb") as f:
            joblib.dump(scaler_y, f)
        
    else:
        # if we only want to test we do not need splits (this is for predefined models). 
        # we can give an specific path for the predefined scaler or use the one predefined. 
        test_feats  = feats
        y_bin_test  = y_bin_full
        if scaler_path is None: 
            with open(scaler_path_base, "rb") as f:
                scaler = joblib.load(f)
            test_scaled = scaler.transform(test_feats)
            with open(scaler_path_base.replace(".pkl", "_y.pkl"), "rb") as f:
                scaler_y = joblib.load(f)
        else: 
            with open(scaler_path, "rb") as f:
                scaler = joblib.load(f)
            test_scaled = scaler.transform(test_feats)
            with open(scaler_path.replace(".pkl", "_y.pkl"), "rb") as f:
                scaler_y = joblib.load(f)
    
    # --- Create sequences ---
    X_train, y_train, y_bin_train_seq = create_sequences(train_scaled, y_bin_train, lookback, forecast) if not testonly else (np.empty((0, lookback, 3)), np.empty((0, forecast)), np.empty((0,)))
    X_val,   y_val,   y_bin_val_seq   = create_sequences(val_scaled, y_bin_val, lookback, forecast) if not testonly else (np.empty((0, lookback, 3)), np.empty((0, forecast)), np.empty((0,)))
    X_test,  y_test,  y_bin_test_seq  = create_sequences(test_scaled, y_bin_test, lookback, forecast)
    
    # --- Weights ---
    w_cl_train =  np.ones(len(y_bin_train_seq), dtype=np.float32) if not testonly else np.empty((0,)) #could be changed to generate weights for class; only in train
    w_cl_val   = np.ones(len(y_bin_val_seq), dtype=np.float32) if not testonly else np.empty((0,))
    w_cl_test  = np.ones(len(y_bin_test_seq), dtype=np.float32)
    
    w_reg_train = compute_seq_weights(y_train, y_bin_train_seq, scaler_y, threshold, forecast=forecast) if not testonly else np.empty((0,)) #only for train
    w_reg_val   = np.ones(len(y_bin_val_seq), dtype=np.float32) if not testonly else np.empty((0,))
    w_reg_test  = np.ones(len(y_bin_test_seq), dtype=np.float32)

    train_std = X_train.std() if not testonly else None

    return (X_train, y_train, y_bin_train_seq, w_cl_train, w_reg_train), (X_val, y_val, y_bin_val_seq, w_cl_val, w_reg_val), \
        (X_test, y_test, y_bin_test_seq, w_cl_test, w_reg_test), scaler, scaler_y, train_std


def create_sequences(feats, y_bin_f, lookback=216, forecast=6):
    '''
    Helper function to compute X, y, y_bin sequences
    Input: 
        - features and binary labels per set
    Output: 
        - sequences: X, y (y_reg) y_bin (y_cl)
    '''
    X, y, y_bin = [], [], []
    n=len(feats)
    for i in range(n - lookback - forecast):
        X.append(feats[i:i+lookback])
        y.append(feats[i+lookback:i+lookback+forecast, 0])  
        y_bin.append(y_bin_f[i + lookback + forecast - 1])  

    return (np.array(X, dtype=np.float32),np.array(y, dtype=np.float32),np.array(y_bin, dtype=int))


def compute_seq_weights(y_reg, y_bin, scaler, threshold=100, forecast=6 ):
    '''
    Helper function to generate w_i weights higher to 1 in early steops of transition 0 --> 1: ensure no fan activation delay
    Input: 
        - binary label sequence of a set
    Output: 
        - weights per instance of the set
    '''
    y_orig = scaler.inverse_transform(y_reg) # original values
    last_vals = y_orig[:, -1]  # based on last value of the forecast horizon
    weights= np.ones_like(y_bin.reshape(-1), dtype=float)
    seq_multiplier = np.ones_like(weights, dtype=float)
    y_bin_flat = y_bin.reshape(-1).astype(int)

    count = 0
    for i in range(len(y_bin_flat)):
        if y_bin_flat[i] == 1:
            if i == 0 or y_bin_flat[i - 1] == 0:
                count = forecast
            else:
                count = max(count - 1, 1)
            seq_multiplier[i] = count**2  
        else:
            count = 0

    weights *= seq_multiplier

    return weights


def create_dataloaders(X_train, y_train, y_train_bin, w_cl_train, w_reg_train, X_val, y_val, y_val_bin, w_cl_val, w_reg_val, X_test, y_test, y_test_bin,  w_cl_test, w_reg_test,
                      scaler, train_std, batch_size=32, lookback=216, dataaug=True, shuffle=True):
    '''
    Function called after prerpocessing data to generate the Dataloaders. To do so we used the RadonDataset
    Input: 
        - preprocess_data output
    Output: 
        - dataloaders
    '''
    train_dataset = RadonDataset(X_train, y_train, y_train_bin, w_cl_train, w_reg_train, scaler=scaler, augment=dataaug, train_std=train_std)
    val_dataset = RadonDataset(X_val, y_val, y_val_bin, w_cl_val, w_reg_val, scaler=scaler, augment=False)  # NO AUGMENT
    test_dataset = RadonDataset(X_test, y_test, y_test_bin, w_cl_test, w_reg_test, scaler=scaler, augment=False)  # NO AUGMENT
                          
    #sampler = WeightedRandomSampler(weights=w_reg_train, num_samples=len(w_reg_train), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False) if len(train_dataset) > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False) if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)  # NO SHUFFLE; B=1
    
    return train_loader, val_loader, test_loader
