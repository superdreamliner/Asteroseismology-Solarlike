import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import medfilt, correlate
import scipy.optimize as op
from scipy.ndimage.filters import percentile_filter
from astropy.timeseries import LombScargle
from astropy.io import fits
import emcee
import corner
import os

def smooth_wrapper(x, y, window_width, window_type="bartlett", samplinginterval=None):

    '''
    Smooth a signal using convolution with a chosen window.
    https://scipy.github.io/old-wiki/pages/Cookbook/SignalSmooth

    Parameters
    ----------
    x : np.ndarray
        Input x-axis values (must be sorted).
    y : np.ndarray
        Signal values corresponding to `x`.
    window_width : float
        Width of the smoothing window in the same units as `x`.
    window_type : {"flat", "hanning", "hamming", "bartlett", "blackman"}, optional
        Type of smoothing window (default: "bartlett").
        "flat" uses a moving average.
    samplinginterval : float, optional
        Resampling interval. If None, estimated from the median spacing in `x`.

    Returns
    -------
    np.ndarray
        Smoothed `y` values aligned with input `x`.
    '''

    if samplinginterval is None:
        samplinginterval = np.median(np.diff(x))

    if not window_type in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError("Unsupported window type. Choose from [flat, hanning, hamming, bartlett, blackman].")

    xp = np.arange(np.min(x), np.max(x), samplinginterval)
    yp = np.interp(xp, x, y)

    window_len = max(3, int(window_width/samplinginterval))
    if window_len % 2 == 0:
        window_len = window_len + 1

    if window_type == "flat":
        w = np.ones(window_len, dtype=float)
    else:
        w = eval("np." + window_type + "(window_len)") 
    
    ys = np.convolve(w / w.sum(), yp, mode="same")

    return np.interp(x, xp, ys)


def simple_smooth(x, window_len, window_type):

    if window_type == "flat":
        w = np.ones(window_len, dtype=float)
    else:
        w = eval("np." + window_type + "(window_len)") 
    y = np.convolve(w / w.sum(), x, mode="same")

    return y


def read_fits(filename):

    flux_keys = ['PDCSAP_FLUX', 'KSPSAP_FLUX', 'DET_FLUX']
    with fits.open(filename, memmap=True) as hdul:
        data = hdul[1].data
        time = data['TIME']
        quality_flag = data['QUALITY']
        flux_key = next((key for key in flux_keys if key in data.columns.names), None)
        if flux_key is None:
            raise KeyError(f"None of the expected flux columns {flux_keys} found in {filename}.")
        flux = data[flux_key]

    return time, flux, quality_flag
    

def lightcurve_prep(time, flux, flag, highpass_window=2.0, outlier_corr=5.0):

    '''
    Prepare a light curve by removing bad points, correcting outliers, and
    applying a high-pass filter (Garcia et al. 2011 method).

    Parameters
    ----------
    time : np.ndarray
        Time array in days.
    flux : np.ndarray
        Raw light curve flux values.
    flag : np.ndarray
        Quality flag array (0 = good).
    highpass_window : float, optional
        High-pass filter width in days. Default is 2.0.
    outlier_corr : float, optional
        Sigma threshold for outlier rejection. Default is 5.0.

    Returns
    -------
    time_day : np.ndarray
        Time array (days) after cleaning.
    time_sec : np.ndarray
        Time array (seconds) after cleaning.
    flux_new : np.ndarray
        Corrected and high-pass filtered flux.
    '''

    time, flux = time[flag == 0], flux[flag == 0]

    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]

    mean_flux, sigma_flux = np.mean(flux), np.std(flux)
    mask = np.abs(flux - mean_flux) <= outlier_corr * sigma_flux
    time, flux = time[mask], flux[mask]

    time_sec = time * 24.0 * 3600.0
    window_sec = highpass_window * 24.0 * 3600.0

    dt = np.diff(time_sec)
    dt_med = np.nanmedian(dt)
    kernelsize = max(3, int(window_sec / dt_med))
    if kernelsize % 2 == 0:
        kernelsize = kernelsize + 1
    flux_medfilt = medfilt(flux, kernel_size=kernelsize)

    mask = flux_medfilt != 0.0
    time_day, time_sec = time[mask], time_sec[mask]
    flux_new = flux[mask] / flux_medfilt[mask]

    return time_day, time_sec, flux_new


def lightcurve(data_dir, cut_freq=2.0*11.57, outlier_corr=5.0, plot_flag=0, starid=None, dirname=None):

    '''
    Correct & Concatenate & Plot the lightcurve.

    Input:
        data_dir: str
            where are the offical .fits LC files

        cut_freq: float
            the cut-off frequency of high-pass filter (μHz).

        outlier_corr: float
            the threshold value of differ correct. (*sigma).

        if plot_flag == 1: 
            lc_after_process.png, only suitable for tess 2-min data, little bit slow.
    
    Output:
        all_time_sec: np.array (second)
            the time after correction.
        all_flux: np.array
            the relative flux after correction. 
    '''

    highpass_window = (1 / cut_freq) * 11.57
    
    all_time_day, all_time_sec, all_flux = [], [], []

    filelist, sector = [], []
    for filename in os.listdir(data_dir):
        if filename.endswith('.fits'):
            filelist.append(filename)
            sector.append(int(filename.split('-')[0][-2:]))
    args = np.argsort(sector)
    filelist = np.array(filelist)[args]

    if plot_flag == 1:
        plt.figure(figsize=(9, 3), dpi=200)
        plt.xlabel('BJTD [BJD-2457000]')
        plt.ylabel('Normalized Flux')
        plt.title(starid + ' Processed Lightcurve')
        time_start, time_end = [], []

    for i in range(len(filelist)):
        time, flux, flag = read_fits(os.path.join(data_dir, filelist[i]))
        time_day, time_sec, flux_new = lightcurve_prep(time, flux, flag, highpass_window=highpass_window, outlier_corr=outlier_corr)
        all_time_day = np.append(all_time_day, time_day)
        all_time_sec = np.append(all_time_sec, time_sec)
        all_flux = np.append(all_flux, flux_new)
        if plot_flag == 1:
            plt.scatter(time_day, flux_new, marker='o', color='black', alpha=1, s=0.5, edgecolor='black', linewidths=0.01)
            time_start.append(time_day[0])
            time_end.append(time_day[-1])
        
    if plot_flag == 1:
        if 1-np.min(all_flux) > np.max(all_flux)-1:
            ylim = 1-np.min(all_flux)
        else:
            ylim = np.max(all_flux)-1
        for i in range(len(filelist)):
            filename = filelist[i]
            t1, t2 = time_start[i], time_end[i]
            plt.vlines(x=t1, ymin=1-ylim, ymax=1+ylim, colors='black', linestyle='--')
            plt.text(t1+1, ylim*0.97+1, s='S'+str(sector[i]), 
                horizontalalignment='left', verticalalignment='top')
            if sector[i]<sector[-1] and sector[i+1]-sector[i]>1:
                for j in range(sector[i+1]-sector[i]-1):
                    plt.text(t2+1+29*j, ylim*0.97+1, s='S'+str(sector[i]+j+1), 
                        horizontalalignment='left', verticalalignment='top')
                    plt.vlines(x=t2+29*j, ymin=1-ylim, ymax=1+ylim, colors='black', linestyle='--')
        for j in range(sector[0]-1):
            plt.text(time_start[0]-29*(sector[0]-1)+29*j+1, ylim*0.97+1, s='S'+str(j+1), 
                horizontalalignment='left', verticalalignment='top')
            plt.vlines(x=time_start[0]-29*(sector[0]-1)+29*j, ymin=1-ylim, ymax=1+ylim, colors='black', linestyle='--')
        for j in range(13-sector[-1]):
            plt.text(time_end[-1]+29*(j)+1, ylim*0.97+1, s='S'+str(sector[-1]+j+1), 
                horizontalalignment='left', verticalalignment='top')
            plt.vlines(x=time_end[-1]+29*(j), ymin=1-ylim, ymax=1+ylim, colors='black', linestyle='--')
        plt.xlim(time_start[0]-29*(sector[0]-1), time_end[-1]+29*(13-sector[-1]))
        plt.ylim(1-ylim, 1+ylim)
        plt.tight_layout()
        plt.savefig(os.path.join(dirname, f'{starid}_lc_processed.png'))
        plt.close()

    average, sigma = np.mean(all_flux), np.std(all_flux)
    all_time_day = all_time_day[np.abs(all_flux - average) <= outlier_corr*sigma]
    all_time_sec = all_time_sec[np.abs(all_flux - average) <= outlier_corr*sigma]
    all_flux = all_flux[np.abs(all_flux - average) <= outlier_corr*sigma]
    
    return all_time_sec, all_flux


def lightcurve_to_psd(time, relative_flux, cut_freq, nyquist, starid=None, dirname=None):

    '''
    Calculate the power spectrum density by Lomb-Scargle periodogram.

    Input:

        time: np.array (second)

        relative_flux: np.array
            the lightcurve after pre-processing.
        
        cut_freq: float
            the cut-off frequency of high-pass filter (μHz).
        
        nyquist: float
            nyquist frequency (μHz).

    Output:

        frequency: np.array (μHZ)
        power_density: np.array (ppm^2)
        psd_smooth: np.array (ppm^2)
    '''

    fnyq_hz = nyquist * 1e-6
    fmin_hz = cut_freq * 1e-6
    baseline = np.nanmax(time) - np.nanmin(time)
    df = 1.0 / baseline
    frequency_hz = np.arange(fmin_hz, fnyq_hz, df)
    power = LombScargle(time, relative_flux).power(frequency_hz, normalization = "psd")
    dt_med = np.nanmedian(np.diff(time))
    power_density = power * dt_med * 1e6
    frequency = frequency_hz * 1e6

    frequency = frequency[power_density > 0.0]
    power_density = power_density[power_density > 0.0]

    if starid and dirname:
        np.savetxt(os.path.join(dirname, f'{starid}_power.csv'),
                   np.column_stack([frequency, power_density]),
                   delimiter=',', header="frequency (μHz), power_density (ppm^2)", encoding="utf-8")
   
    return frequency, power_density


def bg_estimate(frequency, psd, percentile = 40, min_filter_window = 5):
    
    '''
        Crude background estimation based on the psd or smoothed psd.
        Method: scipy.ndimage.filters.percentile_filter.
        Only will be used before collapsed ACF.

    Input:

        frequency, psd: power spectrum density
        
        percentile: 
            percentile filter parameter
        
        interval: int
            step width for speeding up (the number of points)

        min_filter_window: int
            minimum width of filter smooth window

    Output:

        crude_background: np.array (ppm^2)
    '''

    interval = int(len(frequency) / 15000) + 1
    f = frequency[0:len(frequency):interval]
    a = psd[0:len(psd):interval]
    sinterval = np.median(f[1:]-f[:-1])
    power_bg = np.zeros(len(f))
    bounds = np.logspace(np.log10(f.min()), np.log10(f.max()), 30)
    for i in range(len(bounds)-1):
        idx = (f >= bounds[i]) & (f <= bounds[i+1])
        numax_guess = (bounds[i]+bounds[i+1])/2
        dnu_guess = (numax_guess/3050)**0.77 * 135.1
        if int(dnu_guess/sinterval*2) < min_filter_window:
            footprint = min_filter_window
        else:
            footprint = int(dnu_guess/sinterval*2)
        power_bg[idx] = percentile_filter(a, percentile, footprint)[idx]

    crude_background = np.interp(frequency, f, power_bg)

    return crude_background


def auto_correlate(x, y, need_interpolate=False, samplinginterval=None):

    '''
    Generate autocorrelation coefficient as a function of lag. 

    np.correlate perform slowly in large arrays. 

    scipy.signal.correlate is faster and preferable (use FFT to compute the convolution).

    Input:

        x: np.array, the independent variable of the time series.
        y: np.array, the dependent variable of the time series.
        need_interpolate: True or False. True is for unevenly spaced time series.
        samplinginterval: float. Default value: the median of intervals.

    Output:

        lagn: time lag.
        rhon: autocorrelation coeffecient.
    '''

    if len(x) != len(y): 
        raise ValueError("x and y must have equal size.")

    if need_interpolate:
        if samplinginterval is None:
            samplinginterval = np.median(np.diff(x))
        xp = np.arange(np.min(x), np.max(x), samplinginterval)
        yp = np.interp(xp, x, y)
        x, y = xp, yp

    new_y = y - np.mean(y)
    aco = correlate(new_y, new_y, mode='full', method='fft')
    n = len(new_y)
    mid = n - 1
    ac_pos = aco[mid: mid + n]
    lagn = x[:n] - x[0]
    rhon = ac_pos / np.var(y) / n
    rhon = rhon / np.max(rhon)

    return lagn, rhon


def autocorrelate(frequency, power, numax, window_width=250.0, frequency_spacing=None):

    if frequency_spacing is None:
        frequency_spacing = np.median(np.diff(frequency))

    spread = int(window_width/2/frequency_spacing) # Find the spread in indices
    x = int(numax / frequency_spacing) # Find the index value of numax
    x0 = int((frequency[0]/frequency_spacing)) # Transform in case the index isn't from 0
    xt = x - x0
    p_sel = power[xt-spread : xt+spread] # Make the window selection
    p_sel -= np.nanmean(p_sel) # Make it so that the selection has zero mean.

    corr = np.correlate(p_sel, p_sel, mode = 'full')[len(p_sel)-1:] # Correlated the resulting SNR space with itself
    return corr


def estimate_numax_acf2d(frequency, power, psd_smooth=None, plot_flag=1, log_plot=0, starid=None, dirname = None):

    window_width = np.max(frequency)/20
    if np.max(frequency)<300:
        spacing = 2.0
    elif np.max(frequency)>=300 and np.max(frequency)<1000:
        spacing = 5.0
    else:
        spacing = 10.0
    
    numaxs = np.arange(np.min(frequency) + window_width/2, np.max(frequency) - window_width/2, spacing)
    fs = np.median(np.diff(frequency))

    metric = np.zeros(len(numaxs))
    acf2d = np.zeros([int(window_width/2/fs)*2,len(numaxs)])
    for idx, numax in enumerate(numaxs):
        acf = autocorrelate(frequency, power, numax, window_width = window_width, frequency_spacing = fs) # Return the acf at this numax
        acf2d[:, idx] = acf # Store the 2D acf
        metric[idx] = (np.sum(np.abs(acf)) - 1 ) / len(acf) # Store the max acf power normalised by the length

    if len(numaxs) > 10:
        metric_smooth = simple_smooth(metric, window_len=15, window_type='hanning')
    else:
        metric_smooth = metric

    numax_estimate = numaxs[np.argmax(metric_smooth)]
    max_value = np.max(metric)
    min_value = np.min(metric)

    if plot_flag == 1:

        fig, ax = plt.subplots(3, sharex = True, figsize = (12, 12))
        plt.subplots_adjust(hspace = 0, wspace = 0)

        ax[0].set_title(starid+' collapsed ACF', fontsize = 15)
        ax[0].plot(frequency, power, linewidth = '1.0', color = 'lightgray')
        if not psd_smooth is None:
            ax[0].plot(frequency, psd_smooth, linewidth = '1.0', color = 'red')
        label = str(round(numax_estimate, 2)) + ' μHz'
        ax[0].axvline(numax_estimate, c = 'r', linewidth = '2.0', alpha = 0.5, label = label)
        ax[0].legend(loc = 'upper right', fontsize = 15)
        ax[0].text(0.05, 0.9, 'Power Spectrum', horizontalalignment = 'left', transform = ax[0].transAxes, fontsize = 13)
        ax[0].set_ylabel('Power Spectrum', fontsize = 15)

        vmin = np.nanpercentile(acf2d, 5)
        vmax = np.nanpercentile(acf2d, 95)
        ax[1].pcolormesh(numaxs, np.linspace(0, window_width, num = acf2d.shape[0]), acf2d, 
            cmap = 'Greys', vmin = vmin, vmax = vmax)
        ax[1].set_ylabel('Frequency Lag', fontsize = 15)
        ax[1].axvline(numax_estimate, c = 'r', linewidth = '2.0', alpha = 0.5)
        ax[1].text(0.05, 0.9, '2D AutoCorrelation', horizontalalignment = 'left', transform = ax[1].transAxes, fontsize = 13)

        ax[2].plot(numaxs, metric)
        ax[2].plot(numaxs, metric_smooth, linewidth = '2.0', alpha = 0.7, label='Smoothed Metric')
        ax[2].set_xlabel('Frequency (μHz)', fontsize = 15)
        ax[2].set_ylabel('Correlation Metric', fontsize = 15)
        ax[2].axvline(numax_estimate, c = 'r', linewidth = '2.0', alpha = 0.5)
        ax[2].legend(loc = 'upper right', fontsize = 15)
        ax[2].text(0.05, 0.9, 'Correlation Metric', horizontalalignment = 'left', transform = ax[2].transAxes, fontsize = 13)
        ax[2].set_xlim(numaxs[0], numaxs[-1])

        plt.savefig(dirname+starid+'findex.png')
        plt.close()

        if log_plot == 1:

            fig, ax = plt.subplots(3, sharex = True, figsize = (12, 12))
            plt.subplots_adjust(hspace = 0, wspace = 0)

            ax[0].set_title(starid+' collapsed ACF', fontsize = 15)
            ax[0].plot(frequency, power, linewidth = '1.0', color = 'lightgray')
            if not psd_smooth is None:
                ax[0].plot(frequency, psd_smooth, linewidth = '1.0', color = 'red')
            label = str(round(numax_estimate, 2)) + ' μHz'
            ax[0].axvline(numax_estimate, c = 'r', linewidth = '2.0', alpha = 0.5, label = label)
            ax[0].legend(loc = 'upper right', fontsize = 15)
            ax[0].text(0.05, 0.9, 'Power Spectrum', horizontalalignment = 'left', transform = ax[0].transAxes, fontsize = 13)
            ax[0].set_ylabel('Power Spectrum', fontsize = 15)
            ax[0].set_xscale('log')
            ax[0].set_yscale('log')

            vmin = np.nanpercentile(acf2d, 5)
            vmax = np.nanpercentile(acf2d, 95)
            ax[1].pcolormesh(numaxs, np.linspace(0, window_width, num = acf2d.shape[0]), acf2d, 
                cmap = 'Greys', vmin = vmin, vmax = vmax)
            ax[1].set_ylabel('Frequency Lag', fontsize = 15)
            ax[1].axvline(numax_estimate, c = 'r', linewidth = '2.0', alpha = 0.5)
            ax[1].text(0.05, 0.9, '2D AutoCorrelation', horizontalalignment = 'left', transform = ax[1].transAxes, fontsize = 13)
            ax[1].set_xscale('log')

            ax[2].plot(numaxs, metric)
            ax[2].plot(numaxs, metric_smooth, linewidth = '2.0', alpha = 0.7, label='Smoothed Metric')
            ax[2].set_xlabel('Frequency (μHz)', fontsize = 15)
            ax[2].set_ylabel('Correlation Metric', fontsize = 15)
            ax[2].axvline(numax_estimate, c = 'r', linewidth = '2.0', alpha = 0.5)
            ax[2].legend(loc = 'upper right', fontsize = 15)
            ax[2].text(0.05, 0.9, 'Correlation Metric', horizontalalignment = 'left', transform = ax[2].transAxes, fontsize = 13)
            ax[2].set_xlim(numaxs[0], numaxs[-1])
            ax[2].set_xscale('log')

            plt.savefig(dirname+starid+'findex_log.png')
            plt.close()

    return numax_estimate, max_value, min_value


def psd_model(frequency, parameters, nyquist, gran_num=2, type='withgaussian'):
    
    '''
    Compute the power spectral density model.

    Parameters
    ----------
    frequency : np.ndarray
        Frequency array in μHz.
    parameters : array-like
        Model parameters. Expected formats:

        If n_granulation=2:
            [w, a2, b2, a3, b3, height, numax, sigma, c]

        If n_granulation=1:
            [w, a2, b2, height, numax, sigma, c]

    nyquist : float
        Nyquist frequency in μHz.
    n_granulation : int, optional
        Number of granulation components (1 or 2). Default is 2.
    gaussian_mode : {"withgaussian", "withoutgaussian"}, optional
        Whether to include the Gaussian envelope. Default is "withgaussian".

    Returns
    -------
    np.ndarray
        Power spectrum density values.
    '''

    zeta = 2.0 * np.sqrt(2.0) / np.pi  
    part = (np.pi / 2.0) * frequency / nyquist
    sinc2 = (np.sin(part) / part) ** 2.0

    if gran_num == 2:
        w, a2, b2, a3, b3, height, numax, sigma, c = parameters
        gran2 = zeta * a2**2 / (b2 * (1 + (frequency / b2) ** c))
        gran3 = zeta * a3**2 / (b3 * (1 + (frequency / b3) ** c))
        gaussian = height * np.exp(-1.0 * (numax-frequency)**2 / (2.0*sigma**2.0))
        if type == "withgaussian":
            power = sinc2 * (gran2 + gran3 + gaussian) + w
        elif type == "withoutgaussian":
            power = sinc2 * (gran2 + gran3) + w
    
    elif gran_num == 1:
        w, a2, b2, height, numax, sigma, c = parameters
        gran2 = zeta * a2**2 / (b2 * (1 + (frequency / b2) ** c))
        gaussian = height * np.exp(-1.0 * (numax-frequency)**2 / (2.0*sigma**2.0))
        if type == "withgaussian":
            power = sinc2 * (gran2 + gaussian) + w
        elif type == "withoutgaussian":
            power = sinc2 * gran2 + w
    
    return power


def lnlike(parameters, frequency, power, nyquist, gran_num=2, type="withgaussian"):

    '''
    Likelihood function for the power spectral density model.

    Some notes:

        1. the form of the function: ln(p|θ,M) = -Σ(lnM(θ)+D/M(θ))
        2. M: my model, θ: parameters in the model, D: observed data
        3. D/M(θ) is the residual error between true data and the model
        4. the residual error obey chi-square distribution (n=2)
    '''

    powerx = psd_model(frequency=frequency, parameters=parameters, 
        nyquist=nyquist, gran_num=gran_num, type=type)
    like_function = -1.0 * (np.sum(np.log(powerx) + (power/powerx)))

    return like_function


def guess_background_parameters(frequency, psd, psd_smooth, numax_guess, starid=None, dirname=None):
    
    '''
    part-I - guess the power spectrum parameters.
    '''

    # b1_solar = 24.298031575
    b2_solar = 735.4653975
    b3_solar = 2440.5672465 # sometimes tau = 1/b
    numax_solar = 3050
    dnu_guess = (numax_guess / numax_solar) ** 0.77 * 135.1
    zeta = 2.0 * 2.0 ** 0.5 / np.pi

    # to guess the numax peak height
    def FindClosestPoint(x, y, value): 
        index = np.where(np.absolute(x-value) == np.min(np.absolute(x-value)))[0]
        return y[index][0]

    # b1 = numax_guess / numax_solar * b1_solar
    # a1 = (FindClosestPoint(frequency, psd_smooth,b1) * 2 / zeta * b1) ** 0.5
    b2 = numax_guess / numax_solar * b2_solar
    a2 = (FindClosestPoint(frequency, psd_smooth, b2) * 2 / zeta * b2) ** 0.5
    b3 = numax_guess / numax_solar * b3_solar
    a3 = (FindClosestPoint(frequency, psd_smooth, b3) * 2/ zeta * b3) ** 0.5
    h = FindClosestPoint(frequency, psd_smooth, numax_guess) 
    sig = 2.0 * dnu_guess
    w = np.mean(psd_smooth[int(len(frequency)*0.9):]) # white noise
    c = 4.0 # the power degree

    parameters = np.array([w, a2, b2, a3, b3, h, numax_guess, sig, c])
    param_names = ["w", "a2", "b2", "a3", "b3", "h", "numax_guess", "sig", "c"]
    if starid and dirname:
        np.savetxt(
            os.path.join(dirname, f"{starid}_parameters_Guess.csv"),
            parameters.reshape(1, -1),
            delimiter = ",",
            fmt = ["%10.4f"] * len(parameters),
            header = ", ".join(param_names))

    return parameters


def fitting_MLE(frequency, psd, initial_paras, nyquist, gran_num=2, starid=None, dirname=None):
    
    '''
    part-II - fit the power spectrum by Maximum Likelihood Estimation.
    '''

    nll = lambda *args: -lnlike(*args)
    w, a2, b2, a3, b3, height, numax, sigma, c = initial_paras

    if gran_num == 2:
        param_names = ["w", "a2", "b2", "a3", "b3", "h", "numax", "sig", "c"]
        x0 = initial_paras
        bounds = ((0.5*w, 2.0*w), (None, None), (None, None),
                  (0.01, None), (None, None), (0.01, None),
                  (0.5*numax, 1.5*numax), (0.05*numax, 1.0*numax), (None, 10)) 
    elif gran_num == 1:
        param_names = ["w", "a2", "b2", "h", "numax", "sig", "c"]
        x0 = np.delete(initial_paras, [3, 4])
        bounds = ((0.5*w, 2.0*w), (None, None), (None, None),
                  (0.01, None), (0.5*numax, 1.5*numax),
                  (0.05*numax, 1.0*numax), (None, 10)) 
    else:
        raise ValueError(f"Unsupported gran_num={gran_num}. Must be 1 or 2.")
    
    result = op.minimize(nll, x0, args=(frequency, psd, nyquist, gran_num, "withgaussian"), bounds=bounds)
    parameters_MLE = result["x"]

    if starid and dirname:
        np.savetxt(
            os.path.join(dirname, f"{starid}_parameters_MLE.csv"),
            parameters_MLE.reshape(1, -1),
            delimiter=",",
            fmt = ["%10.4f"] * len(parameters_MLE),
            header = ", ".join(param_names))

    return parameters_MLE


def fitting_MCMC(frequency, psd, parameters_MLE, nyquist, bound=0.5, nsteps=1000, nwalkers=20, gran_num=2, starid=None, dirname=None):

    '''
    part-III - fit the power spectrum by MCMC. Recommend to run this part after MLE fitting.
    '''

    def lnprior(parameters):

        '''
        Compute the log-prior for the given parameters.

        The prior is assumed to be uniform, bounded around the
        maximum-likelihood estimates (parameters_MLE) within (1 ± bound) * 100%.

        Returns 1.0 if parameters lie within bounds, -np.inf otherwise.
        '''

        if gran_num == 2:
            param_indices = range(9)  # w, a2, b2, a3, b3, height, numax, sigma, c
        elif gran_num == 1:
            param_indices = range(7)  # w, a2, b2, height, numax, sigma, c
        else:
            return -np.inf

        for index, p in zip(param_indices, parameters):
            min_val = parameters_MLE[index] * (1 - bound)
            max_val = parameters_MLE[index] * (1 + bound)
            if not (min_val < p < max_val):
                return -np.inf

        return 1.0

    def lnprob(parameters, frquency, power, nyquist, gran_num,type):

        '''
        Define the full log-probability function by combining lnlike and lnprior from above.
        '''

        lp = lnprior(parameters)

        if not np.isfinite(lp):
            return -np.inf
        
        return lp + lnlike(parameters, frquency, power, nyquist, gran_num, type)
    
    ndim = len(parameters_MLE)
    pos = parameters_MLE * (1 + 1e-3 * np.random.randn(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(frequency, psd, nyquist, gran_num, "withgaussian"))
    sampler.run_mcmc(pos, nsteps, progress=True)
    samples = sampler.get_chain(discard=250, flat=True)
    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    result_mcmc = np.array([(median, upper-median, median-lower) for lower, median, upper in percentiles.T])
    
    if starid and dirname:
        np.savetxt(os.path.join(dirname, f"{starid}_parameters_MCMC.csv"), 
                   result_mcmc.reshape((ndim, 3)), delimiter=",",
                   fmt=("%10.4f","%10.4f","%10.4f"), header="parameter_value, upper_errorbar, lower_errorbar")
        if gran_num == 2:
            para_label = [r'$w$', r'$a_2$', r'$b_2$', r'$a_3$', r'$b_3$', r'$h$', r'$\nu_{\mathrm{max}}$', r'$\sigma$', r'$c$']
        elif gran_num == 1:
            para_label = [r'$w$', r'$a_2$', r'$b_2$', r'$h$', r'$\nu_{\mathrm{max}}$', r'$\sigma$', r'$c$']
        fig = corner.corner(samples, labels=para_label, quantiles=[0.16, 0.5, 0.84], truths=result_mcmc[:, 0], 
                            show_titles=True, title_kwargs={'fontsize': 15}, label_kwargs={'fontsize': 20})
        fig.savefig(os.path.join(dirname, f'{starid}_corner.png'))
        plt.close()

    return result_mcmc


def plot_with_fit(frequency, psd, psd_smooth=None, parameters=None, nyquist=None, gran_num=2, type='withgaussian', starid=None, dirname=None):

    '''
    Plot the psd including smoothed psd and fitting result.

    gran_num=2: parameters = ['w', 'a2', 'b2', 'a3', 'b3', 'h', 'numax', 'sig', 'c']

    gran_num=1: parameters = ['w', 'a2', 'b2', 'h', 'numax', 'sig', 'c']
    '''

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    # ax.set_title(starid, fontsize=20)
    ax.plot(frequency, psd, linewidth=1.0, color='lightgray')

    if not psd_smooth is None:
        ax.plot(frequency, psd_smooth, linewidth=1.5, color='red')

    if not parameters is None:
        ax.plot(frequency, psd_model(frequency, parameters, nyquist=nyquist, gran_num=gran_num, type='withgaussian'), linewidth='2.5', color='blue')
        zeta = 2.0*2.0**0.5 / np.pi
        part = (np.pi / 2.0) * frequency / nyquist
        power0 = (np.sin(part) / part) ** 2.0
        w = np.zeros(np.shape(frequency), dtype=float)
        w = w + parameters[0]
        ax.plot(frequency, w, linewidth=1.5, color='limegreen')
        if gran_num == 2:
            power2 = zeta*parameters[1]**2.0/(parameters[2]*(1+(frequency/parameters[2])**parameters[8]))
            power3 = zeta*parameters[3]**2.0/(parameters[4]*(1+(frequency/parameters[4])**parameters[8]))
            ax.plot(frequency, power0*power2, linewidth=1.5, color='limegreen')
            ax.plot(frequency, power0*power3, linewidth=1.5, color='limegreen')
        elif gran_num == 1:
            power2 = zeta*parameters[1]**2.0/(parameters[2]*(1+(frequency/parameters[2])**parameters[6]))
            ax.plot(frequency, power0*power2, linewidth=1.5, color='limegreen')
        else:
            raise ValueError(f"Unsupported gran_num={gran_num}. Must be 1 or 2.")

    ax.text(0.05, 0.1, fr'$\nu_\mathrm{{max}} = {parameters[-3]:.2f}\ \mathrm{{μHz}}$', 
        transform=ax.transAxes, ha="left", va="center", fontsize=20)
    ax.set_xlabel(r'Frequency $\mathrm{[μHz]}$', fontsize=20)
    ax.set_ylabel(r'Power Spectrum Density $\mathrm{[ppm^2/μHz]}$', fontsize=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.axis([np.min(frequency), np.max(frequency), np.min(psd), np.max(psd)])
    plt.tight_layout()
    plt.savefig(os.path.join(dirname, f"{starid}_psd.png"))
    plt.close()

    return


def plot_without_fit(frequency, psd, psd_smooth=None, bg=None, starid=None, dirname=None):
    
    '''
    Plot the psd without fitting result.

    This will be used to show crude background estimation.
    '''

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    # ax.set_title(starid, fontsize=20)
    ax.plot(frequency, psd, linewidth=1.0, color='lightgray')
    if not psd_smooth is None:
        ax.plot(frequency, psd_smooth, linewidth=1.5, color='red')
    if not bg is None:
        ax.plot(frequency, bg, linewidth=1.5, color='blue')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel(r'Frequency $\mathrm{[μHz]}$', fontsize=20)
    ax.set_ylabel(r'Power Spectrum Density $\mathrm{[ppm^2/μHz]}$', fontsize=20)
    ax.axis([np.min(frequency), np.max(frequency), np.min(psd), np.max(psd)])
    plt.tight_layout()
    plt.savefig(os.path.join(dirname, f"{starid}_power.png"))
    plt.close()
    
    return


def trim(frequency, numax, psd, psd_smooth=None, lowerindnu=3.0, upperindnu=3.0):

    '''
    Trim the power spectrum around a region of interest (νmax ± n*Δν). 
    '''

    dnu = (numax / 3050.0) ** 0.77 * 135.1
    mask = (frequency >= numax - lowerindnu * dnu) & (frequency <= numax + upperindnu * dnu)
    frequency_new, psd_new = frequency[mask], psd[mask]

    if psd_smooth is None:
        return frequency_new, psd_new
    else:
        psd_smooth_new = psd_smooth[mask]
        return frequency_new, psd_new, psd_smooth_new


def get_dnu_ACF(frequency, psd, numax, plot_flag=1, starid=None, dirname=None):
    
    deltamu_guess = (numax/3050)**0.77 * 135.1

    lagn, rhon = auto_correlate(frequency, psd, need_interpolate=True, samplinginterval=None)
    rhon_smoothed = simple_smooth(rhon, window_len=int(len(lagn)/250), window_type='hanning')
    idx1 = np.where((lagn > 0.8 * deltamu_guess) & (lagn < 1.2 * deltamu_guess))[0]
    lagn1, rhon1, rhon_smoothed1 = lagn[idx1], rhon[idx1], rhon_smoothed[idx1]
    idx2 = np.where(rhon_smoothed1 == np.max(rhon_smoothed1))
    value = lagn1[idx2][0]
    
    if starid and dirname:
        if plot_flag == 1:
            plt.figure(figsize=(10, 10))
            ax = plt.subplot(111)
            # ax.set_title(starid + r' $\Delta \nu$', fontsize=20)
            ax.plot(lagn, rhon, color='lightgray', linewidth=1.5, zorder=0)
            ax.plot(lagn, rhon_smoothed, color='red', linewidth=1.5, zorder=20)
            ax.vlines(value, -1.0, 1.0, color='black', linewidth=1.5, linestyle='--', zorder=10)
            ax.text(0.95, 0.9, fr'$\Delta \nu = {value:.2f}\ \mathrm{{μHz}}$', 
                    transform=ax.transAxes, ha="right", va="center", fontsize=20)
            ax.set_xlabel(r'Frequency $\mathrm{[μHz]}$', fontsize=20)
            ax.set_ylabel('Power Spectrum ACF', fontsize=20)
            ax.axis([np.min(lagn), 5*value, np.min(rhon), np.max(rhon_smoothed) + 0.1])
            ax.tick_params(axis='both', which='major', labelsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(dirname, f'{starid}_dnu_ACF.png'))
            plt.close()

    return value


def get_dnu_error(frequency, psd, nyquist, parameters, numax, gran_num=2):

    dnu_result = []
    for i in range(50):
        noise = np.random.chisquare(2, len(frequency))
        power_density2 = psd * noise / np.mean(noise)
        bg2, psd_bg_corrected2 = psd_bg(frequency, power_density2, 
            parameters, nyquist, gran_num=gran_num)
        frequency2, psd2 = trim(frequency, numax=numax, psd=psd_bg_corrected2, 
            lowerindnu=4.0, upperindnu=4.0)
        dnu2 = get_dnu_ACF(frequency2, psd2, numax=numax, plot_flag=0)
        dnu_result.append(dnu2)
    result = np.percentile(dnu_result, [16, 84])
    lower_limit, upper_limit = result[0], result[1]

    return lower_limit, upper_limit


def echelle(freq, power, dnu, offset=0.0, echelletype='single'):

    '''
    Generate an echelle diagram z-map for stellar oscillation analysis.

    Parameters
    ----------
    freq : np.ndarray
        Frequency array (same units as `dnu`).
    power : np.ndarray
        Power spectrum density values corresponding to `freq`.
    dnu : float
        Large frequency separation.
    offset : float, optional
        Horizontal frequency shift (default: 0.0).
    echelletype : {"single", "double"}, optional
        Whether to stack one or two orders side by side (default: "single").

    Returns
    -------
    xn : np.ndarray
        X-axis values (frequency modulo `dnu`).
    yn : np.ndarray
        Y-axis values (frequency stacking across orders).
    z : np.ndarray
        2D array (z-map) of power values suitable for pcolormesh/imshow plotting.
    '''

    freq = freq - offset
    fmin, fmax = np.min(freq), np.max(freq)

    if fmin <= 0.0:
        fmin = 0.0
    else:
        fmin = fmin - (fmin % dnu)

    mask = (freq >= fmin) & (freq <= fmax)
    trimx, trimp = freq[mask], power[mask]

    sampling_interval = np.median(np.diff(trimx)) * 0.1
    xp = np.arange(fmin, fmax + dnu, sampling_interval)
    yp = np.interp(xp, trimx, trimp)

    n_stack = int((fmax - fmin) / dnu)
    n_element = int(np.round(dnu / sampling_interval))

    morerow = 2
    order_edges = np.arange(0, n_stack + 1) * dnu + fmin + offset
    yn = np.repeat(order_edges, 2)[1:-1]

    if echelletype == "single":
        z = np.zeros([n_stack * morerow, n_element])
        xn = np.arange(1, n_element + 1) / n_element * dnu
        for i in range(n_stack):
            for j in range(i*morerow, (i+1)*morerow):
                z[j, :] = yp[n_element*(i): n_element*(i+1)]

    if echelletype == "double":
        z = np.zeros([n_stack*morerow, 2*n_element])
        xn = np.arange(1, 2*n_element+1) / n_element * dnu
        for i in range(n_stack):
            for j in range(i*morerow, (i+1)*morerow):
                z[j,:] = np.concatenate([yp[n_element*(i): n_element*(i+1)], yp[n_element*(i+1): n_element*(i+2)]])

    return xn, yn, z


def plot_echelle(freq, power, dnu, offset=0.0, echelletype='single', starid=None, dirname=None):

    '''
    Plot the echelle diagram for a given amplitude (or power) spectrum.
    '''

    echx, echy, echz = echelle(freq, power, dnu, offset=offset, echelletype=echelletype)
    echz = np.sqrt(echz)
    levels = np.linspace(np.min(echz), np.max(echz), 300)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    if echelletype == "double":
        ax.axvline(dnu, c='r', linewidth=2.0, alpha=0.5)
    # ax.set_title(starid, fontsize=20)
    ax.contourf(echx, echy, echz, levels=levels, cmap='gray_r')
    ax.axis([np.min(echx), np.max(echx), np.min(echy), np.max(echy)])
    ax.set_xlabel(fr'Frequency mod {round(dnu, 2)} $\mathrm{{[μHz]}}$', fontsize=20)
    ax.set_ylabel(r'Frequency $\mathrm{[μHz]}$', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(dirname, f'{starid}_echelle_{echelletype}.png'))
    plt.close()

    return


def psd_bg(frequency, psd, parameters, nyquist, gran_num=2):

    '''
    Calculate the background of power spectrum with identified parameters.

    Input:
        parameters = ['w', 'a2', 'b2', 'a3', 'b3', 'h', 'numax', 'sigma', 'c']

    Output:
        background of power spectrum
        the psd without background
    '''

    bg = psd_model(frequency, parameters, nyquist, gran_num=gran_num, type='withoutgaussian')
    psd_without_bg = psd / bg

    return bg, psd_without_bg


def dnu(frequency, psd, numax, parameters, nyquist, gran_num=2, echelle_plot=1, echelletype='single', starid=None, dirname=None):

    '''
        1. Background correction 2. dnu and its uncertainty calculation 3. plot the echelle diagram

    Output:
        float: dnu_result, lower_limit, upper_limit

        figures: power_without_bg.csv, power_no_bg.png, dnuACF.png, dnuACF.csv
                 echelle_single.png / echelle_double.png
    '''

    bg, psd_bg_corrected = psd_bg(frequency, psd, parameters, nyquist, gran_num=gran_num)
    psd_bg_corrected_smooth = smooth_wrapper(frequency, psd_bg_corrected, window_width=1.0)

    if starid and dirname:
        np.savetxt(os.path.join(dirname, f'{starid}_power_no_bg.csv'),
                   np.column_stack([frequency, psd_bg_corrected]),
                   delimiter=',', header="frequency (μHz), power_density (ppm^2)", encoding="utf-8")

    frequency2, psd2, psd_smooth2 = trim(frequency, numax=numax, psd=psd_bg_corrected, 
        psd_smooth=psd_bg_corrected_smooth, lowerindnu=5.0, upperindnu=5.0)

    if starid and dirname:
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111)
        # ax.set_title(starid, fontsize=20)
        ax.plot(frequency2, psd2, linewidth=1.0, color='lightgray')
        ax.plot(frequency2, psd_smooth2, linewidth=1.5, color='red')
        ax.set_xlabel(r'Frequency $\mathrm{[μHz]}$', fontsize=20)
        ax.set_ylabel('SNR', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.axis([np.min(frequency2), np.max(frequency2), np.min(psd2), 1.5*np.max(psd_smooth2)])
        plt.tight_layout()
        plt.savefig(os.path.join(dirname, f'{starid}_power_no_bg.png'))
        plt.close()

    dnu_result = get_dnu_ACF(frequency2, psd2, numax, starid=starid, dirname=dirname)
    lower_limit, upper_limit = get_dnu_error(frequency, psd, nyquist, parameters, numax, gran_num=gran_num)
    array = np.array([dnu_result, upper_limit - dnu_result, dnu_result - lower_limit])
    if starid and dirname:
        np.savetxt(
            os.path.join(dirname, f"{starid}_dnu_ACF.csv"),
            array.reshape(1, -1),
            delimiter=",",
            fmt = ["%10.4f"] * len(array),
            header = "dnu_result, upper_errorbar, lower_errorbar")
        if echelle_plot == 1:
            plot_echelle(freq=frequency2, power=psd2, dnu=dnu_result, 
                echelletype=echelletype, starid=starid, dirname=dirname)

    return dnu_result, lower_limit, upper_limit
