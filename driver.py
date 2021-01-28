import numpy as np
import main
import os

sep = "\\" if os.name == "nt" else "/"

nyquist = 4166.67
granulation = 2

all_output = 'D:\\test\\output'
all_data = 'D:\\test\\data'

target_list = np.loadtxt('D:\\test\\targetlist.csv', delimiter=',', dtype=str)

for i in range(len(target_list)):

    starid, teff, lum = target_list[i,0], float(target_list[i,1]), float(target_list[i,2])
    # mumax_guess = float(target_list[i,3])
    mumax_guess = ((lum**(-1) * (teff/5777)**5) ** 0.926) * 3050
    print(starid)

    try:

        if os.path.exists(all_output + sep + starid):
            output_dir = all_output + sep + starid + sep
        else:
            os.mkdir(all_output + sep + starid)
            output_dir = all_output + sep + starid + sep

        data_dir = all_data + sep + starid
        time, flux = main.lightcurve(data_dir, cut_freq=2.0*11.57, outlier_corr=5.0, 
            plot_flag=1, starid=starid, dirname=output_dir)
        
        print('----- Lightcurve preparation completed -----')

        frequency, power_density = main.lightcurve_to_psd(time, flux, cut_freq=2.0*11.57, 
            nyquist=nyquist, starid=starid, dirname=output_dir)

        # power_filename = output_dir + starid + 'power.dat'
        # power_file = np.loadtxt(power_filename)
        # frequency, power_density = power_file[:,0], power_file[:,1]
        psd_smooth = main.smooth_wrapper(frequency, power_density, window_width = 3.0)

        print('----- Power spectrum preparation completed -----')

        para_guess = main.guess_background_parameters(frequency=frequency, psd=power_density, 
            psd_smooth=psd_smooth, mumax_guess=mumax_guess, 
            starid=starid, dirname=output_dir)

        para_MLE = main.fitting_MLE(frequency=frequency, psd=power_density, 
            psd_smooth=psd_smooth, initial_paras=para_guess, nyquist=nyquist, 
            gran_num=granulation, starid=starid, dirname=output_dir)

        main.plot_with_fit(frequency=frequency, psd=power_density, 
            psd_smooth=psd_smooth, parameters=para_MLE, nyquist=nyquist, type='withgaussian', 
            gran_num=granulation, starid=starid+'_MLE', dirname=output_dir)
        
        print('----- MLE power fitting completed -----')

        para_MCMC = main.fitting_MCMC(frequency=frequency, psd=power_density, 
            psd_smooth=psd_smooth, parameters_MLE=para_MLE, nyquist=nyquist, 
            bound=0.5, nsteps=3000, nwalkers=20, 
            gran_num=granulation, starid=starid, dirname=output_dir)

        main.plot_with_fit(frequency=frequency, psd=power_density, 
            psd_smooth=psd_smooth, parameters=para_MCMC, nyquist=nyquist, type='withgaussian', 
            gran_num=granulation, starid=starid+'_MCMC', dirname=output_dir)

        print('----- MCMC power fitting completed -----')

        dnu_result, lower_limit, upper_limit = main.dnu(frequency=frequency, psd=power_density, 
            mumax=para_MCMC[-3], parameters=para_MCMC, nyquist=nyquist, 
            gran_num=granulation, echelle_plot=1, echelletype='double',
            starid=starid, dirname=output_dir)

        print('----- Dnu calculation completed -----')
    
    except:

        print('Error')
        continue
