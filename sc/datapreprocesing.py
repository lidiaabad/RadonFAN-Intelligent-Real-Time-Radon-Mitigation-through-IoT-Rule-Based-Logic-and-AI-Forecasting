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
            
            if self.yb[idx]== 1: 
                prob=0.75
                factor=2.5
            else: 
                prob=0.25
                factor=1
                               
            if random.random() < prob:
                noise_scale = factor* 0.05 * self.train_std  # ajusta 0.05–0.2
                noise = np.random.normal(0, noise_scale, size=x.shape)
                x += noise
                #print("jittering with noise", noise)

            if random.random() < prob:
                scale_factor = random.uniform(1-(0.1*factor), 1 + (0.1*factor))
                x *= scale_factor
                #print("scaling with factor", scale_factor)

            #time warping
            if random.random() < prob and len(x) > 3:
                orig_idx = np.arange(len(x))
                random_idx = np.linspace(0, len(x)-1, num=len(x))
                perturb = np.random.uniform(-3, 3, size=len(x))
                perturb_smooth = scipy.ndimage.gaussian_filter1d(perturb, sigma=(1*factor))  # sigma controla suavidad
                warped_idx = np.clip(np.arange(len(x)) + perturb_smooth, 0, len(x)-1)
                f = interp1d(orig_idx, x, axis=0, kind='cubic', fill_value="extrapolate")
                x = f(warped_idx)
                #print("time warping with sigma", factor)
        
            #magniture warping
            if random.random() <prob and len(x)> self.segment_size:
                n_segments = len(x) // self.segment_size
                for i in range(n_segments):
                    start = i * self.segment_size
                    end = start + self.segment_size
                    factor = random.uniform(1-(0.1*factor), 1 + (0.1*factor))  # local amplitude change
                    x[start:end] *= factor
                    #print("magnitude warping with factor", factor)

            if random.random() < 0.1*factor and len(x) > 4: 
                n = random.randint(1,len(x) //4)
                x[-n:] = 0.0 
                #print("masking with len", n)
            
            
        return (torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(yb, dtype=torch.float32), torch.tensor(w_cl, dtype=torch.float32), torch.tensor(w_reg, dtype=torch.float32))

def preprocess_data(file_path, lookback=216, forecast=6, threshold=300, numfeats=1, testonly=False, scaler_path=None):
    
    if forecast==6: 
        margin=1.00
    else: 
        margin=0.9
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

    # --- Compute smoothed Rv ---

    n = len(Rt)
    Rv = np.zeros_like(Rt, dtype=float)
    Rv[0] = Rt[0]
    for t in range(1, n):
        Rv[t] = 0.5 * Rt[t] + 0.5 * Rv[t-1]

    # --- Divergence ---
    Dt = Rt - Rv

    
    n = len(Rt)
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
    dist_to_threshold =  Rt - threshold*margin
    sv = np.zeros_like(slope, dtype=float)
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
        #mean_rt_local = np.mean(slope[t - lookback : t]) + 1e-6
        #lastrelrt_ratio[t] = (Rt[t] - mean_rt_local) / np.abs(mean_rt_local)
        #mean_slope_local = np.mean(slope[t - lookback : t]) + 1e-6
        #lastrelslope_ratio[t] = (slope[t] - mean_slope_local) / np.abs(mean_slope_local)
        
    cumsum_win = np.zeros(n)
    cumsum_win[:lookback+1] = np.sum(Rt[:lookback+1])  # inicial
    for t in range(lookback+1, n):
        cumsum_win[t] = np.sum(Rt[t - lookback + 1 : t + 1])
    
    max_Rt_lookback = np.zeros(n)
    max_Rt_lookback[:lookback+1] = Rt[:lookback+1].max()
    for t in range(lookback+1, n):
        max_Rt_lookback[t] = np.max(Rt[t - lookback + 1 : t + 1])

    count_above_thr = np.zeros(n, dtype=int)
    for t in range(lookback+1, n):
        window = Rt[t - lookback + 1 : t + 1]
        count_above_thr[t] = np.sum(window > (threshold * 0.75))
        
    windows3 = sliding_window_view(Rt, window_shape=3)
    mean_3 = np.mean(windows3, axis=1)
    mean_3 = np.concatenate([[mean_3[0]]*3, mean_3[:-1]])
    windows12 = sliding_window_view(Rt, window_shape=12)
    mean_12 = np.mean(windows12, axis=1)
    mean_12 = np.concatenate([[mean_12[0]]*12, mean_12[:-1]])
    ma_ratio=mean_3 / (mean_12 + 1e-6)
    acel=slope3-slope
    
    
    
        
    #reldistRt=lastrelRt/(dist_to_threshold + 1e-6)
    #relslopeRt=lastrelslope/(dist_to_threshold + 1e-6)
    # --- Combine features ---
    if numfeats==16: 
        feats = np.stack([Rt, Rv, Dt, slope, slope3, var3, slope6, var6, sv, lastrelRt, lastrelslope, cumsum_win, max_Rt_lookback, count_above_thr, ma_ratio, acel], axis=-1) #,  reldistRt, relslopeRt], axis=-1)
    elif numfeats==4: 
        feats = np.stack([Rt, cumsum_win, max_Rt_lookback, count_above_thr], axis=-1) #,lastrelRt, count_above_thr
    else: 
        feats = Rt.reshape(-1, 1)
    
    # ---- Binary Labels computation 
    y_bin_full = np.zeros(n, dtype=int)
    '''
    for i in range(1, n):
        if Rv[i] < 200: # Rv
            y_bin_full[i] = 0
        elif Rv[i] > 300: # Rv
            y_bin_full[i] = 1
        else:
            if slope[i] > 0.3:
                y_bin_full[i] = 1
            elif slope[i] < -0.3:
                y_bin_full[i] = 0
            else:
                y_bin_full[i] = y_bin_full[i-1]
    '''
        
    for i in range(forecast, n):
        if np.any(Rt[i-forecast+1:i+1] >= threshold*margin):
        #if (Rt[i] >= threshold*margin): # and Rt[i-1] < threshold*margin) or (Rt[i] < threshold*margin and Rt[i-1] >= threshold*margin) :
            y_bin_full[i] = 1
        else: 
            y_bin_full[i] = 0
            
    #y_bin_full = smooth_binary_runs(y_bin_full)

    y_bin_full_smooth = smooth_binary_transitions(y_bin_full, ramp_len=0, curve='steady')
    
    # --- Splits ---
    split_test = int(0.15 * n)
    split_val = int(0.30* n)

    # ---- Scaler: create or load and apply it to splits
    base = os.path.basename(file_path)
    name = os.path.splitext(base)[0]
    models_dir = "../models" if file_path.startswith("../data/") else "models"
    scaler_path_base = os.path.join(models_dir, f"{name}_scaler.pkl")

    if not testonly:
        train_feats = feats[split_val:]
        val_feats   = feats[split_test:split_val]
        test_feats  = feats[:split_test]
        '''
        y_bin_train = y_bin_full_smooth[split_val:]
        y_bin_val   = y_bin_full_smooth[split_test:split_val]
        y_bin_test  = y_bin_full_smooth[:split_test]
        '''
        y_bin_train = y_bin_full[split_val:]
        y_bin_val   = y_bin_full[split_test:split_val]
        y_bin_test  = y_bin_full[:split_test]
        
        # Fit scaler on training features (3 columns)
        scaler = RobustScaler()#MinMaxScaler(feature_range=(-1, 1)) #RobustScaler()
        scaler.fit(train_feats)
        os.makedirs(models_dir, exist_ok=True)
        with open(scaler_path_base, "wb") as f:
            joblib.dump(scaler, f)

        train_scaled = scaler.transform(train_feats)
        val_scaled   = scaler.transform(val_feats)
        test_scaled  = scaler.transform(test_feats)
        
        #scaler_y=scaler
        scaler_y = RobustScaler() #MinMaxScaler(feature_range=(-1, 1)) #RobustScaler() 
        scaler_y.fit(train_feats[:, [0]]) #only on Rt
        with open(scaler_path_base.replace(".pkl", "_y.pkl"), "wb") as f:
            joblib.dump(scaler_y, f)
        
        
    else:
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
    
    # ---- Print information on thresholds
    
    values_to_scale = np.array([[threshold*0.25], [threshold*0.85], [threshold*0.9], [threshold*0.95], [threshold], [threshold*2]])  # debe ser 2D: (n_samples, 1)
    scaled_values = scaler_y.transform(values_to_scale)
    for orig, scaled in zip(values_to_scale.flatten(), scaled_values.flatten()):
        print(f"Original y = {orig} -> Scaled y = {scaled}")
    
    # --- Create sequences ---
    X_train, y_train, y_bin_train_seq = create_sequences(train_scaled, y_bin_train, lookback, forecast) if not testonly else (np.empty((0, lookback, 3)), np.empty((0, forecast)), np.empty((0,)))
    X_val,   y_val,   y_bin_val_seq   = create_sequences(val_scaled, y_bin_val, lookback, forecast) if not testonly else (np.empty((0, lookback, 3)), np.empty((0, forecast)), np.empty((0,)))
    X_test,  y_test,  y_bin_test_seq  = create_sequences(test_scaled, y_bin_test, lookback, forecast)
    
    def count_transitions(arr, name="array"):
        """Count 0→1 transitions along a 1D array (binary)."""
        arr = np.asarray(arr).ravel().astype(int)
        transitions = np.where((arr[1:] == 1) & (arr[:-1] == 0))[0] + 1
        print(f"{name}: {len(transitions)} transitions (0→1)")
        transitions_n = np.where((arr[1:] == 0) & (arr[:-1] == 1))[0] + 1
        print(f"{name}: {len(transitions_n)} transitions (1→0)")
        return transitions

    # --- (A) transitions in sequence-based binaries ---
    count_transitions(y_bin_train_seq, "y_bin_train_seq")
    count_transitions(y_bin_val_seq,   "y_bin_val_seq")
    count_transitions(y_bin_test_seq,  "y_bin_test_seq")

    # --- Weights ---
    w_cl_train = compute_seq_weights(y_bin_train_seq, forecast=forecast) if not testonly else np.empty((0,))
    w_cl_val   = np.ones(len(y_bin_val_seq), dtype=np.float32) if not testonly else np.empty((0,))
    w_cl_test  = np.ones(len(y_bin_test_seq), dtype=np.float32)
    
    w_reg_train = compute_sampler_weights(y_train, y_bin_train_seq, scaler_y, threshold, margin, forecast=forecast) if not testonly else np.empty((0,))
    w_reg_val   = np.ones(len(y_bin_val_seq), dtype=np.float32) if not testonly else np.empty((0,))
    w_reg_test  = np.ones(len(y_bin_test_seq), dtype=np.float32)

    train_std = X_train.std() if not testonly else None

    return (X_train, y_train, y_bin_train_seq, w_cl_train, w_reg_train), \
        (X_val, y_val, y_bin_val_seq, w_cl_val, w_reg_val), \
        (X_test, y_test, y_bin_test_seq, w_cl_test, w_reg_test), \
        scaler, scaler_y, train_std


def create_sequences(feats, y_bin_f, lookback=216, forecast=6):
    '''
    Helper function to compute X, y, y_bin sequences
    Input: 
        - features and binary labels per set
    Output: 
        - sequences
    '''
    X, y, y_bin = [], [], []
    n=len(feats)
    for i in range(n - lookback - forecast):
        X.append(feats[i:i+lookback]) #slope
        y.append(feats[i+lookback:i+lookback+forecast, 0])  # 0: Rt, 1: slope; 2: prop_slope
        y_bin.append(y_bin_f[i + lookback + forecast - 1])  # valor binario en último paso de y
        '''
        if np.any(y_bin_f[i + lookback : i + lookback + forecast] == 1):
            y_bin.append(1)
        else:
            y_bin.append(0)
        '''
    return (np.array(X, dtype=np.float32),np.array(y, dtype=np.float32),np.array(y_bin, dtype=int))

def smooth_binary_runs(y, min_len=3):
    y = y.copy()
    # Encuentra cambios 0→1 o 1→0
    diff = np.diff(np.concatenate(([y[0]], y, [y[-1]])))
    # Índices donde cambian los valores
    change_idx = np.where(diff != 0)[0]
    
    # Longitudes de los bloques
    run_lengths = np.diff(np.concatenate(([0], change_idx, [len(y)])))
    # Valores de cada bloque
    values = []
    start = 0
    for length in run_lengths:
        values.append(y[start])
        start += length
    
    # Corrige bloques cortos
    start = 0
    for val, length in zip(values, run_lengths):
        end = start + length
        if length < min_len:
            if start > 0:
                y[start:end] = y[start - 1]  # iguala al bloque anterior
            elif end < len(y):
                y[start:end] = y[end]       # o al siguiente
        start = end
    return y


def smooth_binary_transitions(y, ramp_len=6, curve='linear'):
    """
    Suaviza transiciones 0→1 y 1→0 en una señal binaria (0/1).
    
    y: array 1D de 0s y 1s
    ramp_len: longitud (en pasos) de la rampa
    curve: tipo de curva ('linear', 'cosine', 'sigmoid')
    """
    y = np.array(y, dtype=float)
    y_smooth = y.copy()
    
    if curve == 'steady':
        diff = np.diff(y)
        on_edges = np.where(diff == 1)[0]  # transiciones 0→1

        for t_on in on_edges:
            y_smooth[t_on-ramp_len:t_on+1] = 1

        return y_smooth

    # --- Precomputar la rampa
    x = np.linspace(0, 1, ramp_len)
    if curve == 'linear':
        ramp = x
    elif curve == 'cosine':
        ramp = 0.5 - 0.5 * np.cos(np.pi * x)
    elif curve == 'sigmoid':
        ramp = 1 / (1 + np.exp(-10*(x - 0.5)))
    else:
        raise ValueError("Unknown curve type")

    # --- Detectar transiciones
    diff = np.diff(y)
    on_edges = np.where(diff == 1)[0]      # 0→1
    off_edges = np.where(diff == -1)[0]    # 1→0

    # --- Aplicar rampas
    for t_on in on_edges:
        # subida: t_on-ramp_len+1 ... t_on (sin pasarse de 0)
        start = max(0, t_on - ramp_len + 1)
        segment = ramp[-(t_on - start + 1):]  # parte final de la rampa
        y_smooth[start:t_on+1] = np.maximum(y_smooth[start:t_on+1], segment)

    for t_off in off_edges:
        # bajada: t_off+1 ... t_off+ramp_len (sin pasarse del final)
        end = min(len(y), t_off + ramp_len + 1)
        segment = ramp[:end - (t_off+1)]
        y_smooth[t_off+1:end] = np.minimum(y_smooth[t_off+1:end], 1 - segment)

    return np.clip(y_smooth, 0, 1)

def compute_seq_weights(yb_seq, forecast=6):
    '''
    Helper function to generate weights higher to 1 in early steops of transition 0 --> 1: ensure no fan activation delay
    Input: 
        - binary label sequence of a set
    Output: 
        - weights per instance of the set
    '''

    T = len(yb_seq)
    weights = np.ones(T, dtype=np.float32)

    decay_factor = 1.0  # initialize decay

    for i in range(1, T):
        if yb_seq[i] == 1:
            # transition 0 -> 1 or start of sequence
            if yb_seq[i-1] == 0:
                decay_factor = forecast + 1
            weights[i] = decay_factor ** 4
            # decay for next consecutive 1
            decay_factor = max(1.0, decay_factor - 1)
        else:
            weights[i] = 1.0
            decay_factor = 1.0  # reset decay after 0

    return weights

def compute_transition_weights(y_bin, boost_weight=10.0, base_weight=0.1, window=6):
    """
    Asigna pesos muy bajos a todo excepto ±window pasos alrededor de transiciones.
    
    Args:
        y_bin: array binario (0 o 1)
        boost_weight: peso multiplicativo alto cerca de transiciones
        base_weight: peso base bajo para los demás puntos
        window: número de pasos antes y después de la transición

    Returns:
        weights: array con los mismos valores que y_bin
    """
    n = len(y_bin)
    weights = np.ones(n) * base_weight

    # Encuentra índices de cambio de estado (0→1 o 1→0)
    transitions = np.where(y_bin[1:] != y_bin[:-1])[0]

    for idx in transitions:
        start = max(0, idx - window)
        end = min(n, idx + window + 1)
        weights[start:end] = boost_weight

    # Normaliza si quieres mantener media = 1 (opcional)
   # weights /= (np.mean(weights) + 1e-8)

    return weights

def compute_sampler_weights(y_reg, y_bin, scaler, threshold=100, margin=0.9, 
                                 low_frac=0.25, boost_weight=10.0, forecast=6,
                                 window=6):

    # --- Desnormalizamos todos los valores
    y_orig = scaler.inverse_transform(y_reg)

    # --- Tomamos SOLO el último valor de cada fila
    last_vals = y_orig[:, -1]  # shape (N,)
    '''
    # --- Parámetros de control
    low_limit = low_frac * threshold
    sigma = low_frac * threshold

    # --- Gaussiana centrada en threshold*0.9 (3-6 pasos antes)
    weights = np.exp(-0.5 * ((last_vals - (threshold*0.9)) / sigma) ** 2)

    # --- Penalización fuerte para valores muy bajos
    low_mask = last_vals < low_limit
    weights[low_mask] *= 0.05

    # --- Refuerzo para valores dentro del rango [0.8, 1.2] * threshold
    near_mask = (last_vals >= 0.5 * threshold) & (last_vals <= 1.5 * threshold)
    boost = 1 + (boost_weight - 1) * np.exp(-0.5 * ((last_vals - (threshold*0.9)) / (0.2 * threshold)) ** 2)
    weights[near_mask] *= boost[near_mask]
    '''
    weights= np.ones_like(y_bin.reshape(-1), dtype=float)
    seq_multiplier = np.ones_like(weights, dtype=float)

    # Convertimos y_bin a 1D si viene por columnas (e.g. (N,1))
    y_bin_flat = y_bin.reshape(-1).astype(int)

    count = 0

    for i in range(len(y_bin_flat)):
        if y_bin_flat[i] == 1:
            if i == 0 or y_bin_flat[i - 1] == 0:
                # Primer 1 después de 0 → forecast
                count = forecast
            else:
                # Segundo, tercero... va bajando
                count = max(count - 1, 1)
            seq_multiplier[i] = count**2  # cuadrado del valor actual
        else:
            # Reinicia la secuencia si hay un 0
            count = 0

    weights *= seq_multiplier

    return weights


def compute_class_weights(y_train, scaler, threshold=100, steepness=0.5, eps=1e-6):
    """
    Pesos altos y planos en la región [lower, upper], decayendo rápido fuera.
    
    Inputs:
        y_train: valores escalados
        scaler: scaler usado para y_train
        lower, upper: límites de la región plana
        steepness: controla qué tan rápido decae fuera de la región
    """
    # Inverse transform
    if y_train.ndim == 1:  # forecast == 1, univariate
        y_train = y_train.reshape(-1, 1)
    print(y_train.shape)
    y_train_original = scaler.inverse_transform(y_train)
    last_point = y_train_original[:, -1]
    #y_train_flat = y_train.reshape(-1, y_train.shape[-1])
    #y_train_original_flat = scaler.inverse_transform(y_train_flat)
    #last_point = y_train_original_flat[:, -1]
    upper= threshold*(0.9+0.2)
    lower= threshold*(0.9-0.2)
    # Función sigmoide para los límites
    left_decay = 1 / (1 + np.exp(-(last_point - lower) / (steepness * np.std(last_point))))
    right_decay = 1 / (1 + np.exp((last_point - upper) / (steepness * np.std(last_point))))

    # Combinamos para tener “mesa” plana en el centro
    raw_weights = left_decay * right_decay
    '''
    # Evitamos varianzas 0
    raw_weights = np.var(y_train, axis=1) 
    raw_weights = np.maximum(raw_weights, eps)
    '''
    
    raw_weights /= np.mean(raw_weights)
    
    return raw_weights


def create_dataloaders(X_train, y_train, y_train_bin, w_cl_train, w_reg_train, X_val, y_val, y_val_bin, w_cl_val, w_reg_val, X_test, y_test, y_test_bin,  w_cl_test, w_reg_test,
                      scaler, train_std, batch_size=32, lookback=216, dataaug=True, shuffle=True):
    
    '''
    Function called after prerpocessing data to generate the Dataloaders. To do so we used the RadonDataset
    Input: 
        - preprocess_data output
    Output: 
        - dataloaders
    '''
    train_dataset = RadonDataset(X_train, y_train, y_train_bin, w_cl_train, w_reg_train, scaler=scaler,augment=dataaug, train_std=train_std)
    val_dataset = RadonDataset(X_val, y_val, y_val_bin, w_cl_val, w_reg_val, scaler=scaler, augment=False)
    test_dataset = RadonDataset(X_test, y_test, y_test_bin, w_cl_test, w_reg_test, scaler=scaler, augment=False)
    
    #sampler = WeightedRandomSampler(weights=w_reg_train, num_samples=len(w_reg_train), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False) if len(train_dataset) > 0 else None
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=False) if len(train_dataset) > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False) if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader

'''
===================================== OTHER HELR FUNCTIONS =======================================
'''

def create_delta_sequences(data, lookback=216, forecast=6):
    print("CREATING DELTA SEQUENCES")
    radon_values = data[:, 1].astype(float)  # select only Radon
    radon_delta = np.diff(radon_values, n=1)  # length n-1
    X, y = [], []
    for i in range(len(radon_delta) - lookback - forecast):
        # X_delta: differences of lookback window
        X.append(radon_delta[i:i + lookback])
        y.append(radon_delta[i + lookback :i + lookback + forecast ])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def inverse_transform(y, scaler):
    # y es numpy array (N, 6)
    N, steps = y.shape
    y_reshape = y.reshape(-1, 1)  # a 2D para scaler
    y_inv = scaler.inverse_transform(y_reshape)
    return y_inv.reshape(N, steps)

def create_binary_labels(y_inv, slope_threshold=0.3, early_threshold=200, main_threshold=300):
    
    X = np.zeros_like(y_inv)
    X[:, 0] = y_inv[:, 0]
    for t in range(1, y_inv.shape[1]):
        X[:, t] = 0.5 * y_inv[:, t] + 0.5 * X[:, t-1]

    # último y penúltimo
    X_last = X[:, -1]
    X_prev = X[:, -2]
    
    # pendiente
    slope = (X_last - X_prev) / 10.0
    y_final = np.zeros(len(X_last), dtype=int)

    for i in range(len(X_last)):
        if X_last[i] < early_threshold:
            y_final[i] = 0
        elif X_last[i] > main_threshold:
            y_final[i] = 1
        else:
            if slope[i] > slope_threshold:
                y_final[i] = 1
            elif slope[i] < -slope_threshold:
                y_final[i] = 0
            else:
                y_final[i] = y_final[i-1]

    return y_final

