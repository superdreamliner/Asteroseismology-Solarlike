import numpy as np
import main
import os

nyquist = 1 / (2 * (30*60)) * 1e6
cutoff_frequency = 1.0 * 11.57
granulation = 2

main_dir = 'D:\\Asteroseismology-Solarlike-main'
all_output = os.path.join(main_dir, 'output')
all_data = os.path.join(main_dir, 'data')

target_list = np.loadtxt(os.path.join(main_dir, 'targetlist.csv'), delimiter=',', dtype=str, ndmin=2)

for i in range(len(target_list)):

    starid, teff, lum = target_list[i, 0], float(target_list[i, 1]), float(target_list[i, 2])
    # numax_guess = float(target_list[i,3])
    numax_guess = ((lum**(-1) * (teff/5777)**5) ** 0.926) * 3050
    print(starid)

    # try:

    output_dir = os.path.join(all_output, starid)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    time, flux = main.lightcurve(data_dir= os.path.join(all_data, starid), 
                                 cut_freq=cutoff_frequency, 
                                 outlier_corr=5.0, plot_flag=0, 
                                 starid=starid, dirname=output_dir)
    
    print('----- Lightcurve preparation completed -----')

    frequency, power_density = main.lightcurve_to_psd(time, 
                                                      flux, 
                                                      cut_freq=cutoff_frequency, 
                                                      nyquist=nyquist, 
                                                      starid=starid, dirname=output_dir)

    # power_file = np.loadtxt(os.path.join(output_dir, f'{starid}_power.dat'))
    # frequency, power_density = power_file[:,0], power_file[:,1]
    psd_smooth = main.smooth_wrapper(frequency, power_density, window_width=3.0)

    print('----- Power spectrum preparation completed -----')

    para_guess = main.guess_background_parameters(frequency=frequency, 
                                                  psd=power_density, 
                                                  psd_smooth=psd_smooth, 
                                                  numax_guess=numax_guess, 
                                                  starid=starid, dirname=output_dir)

    para_MLE = main.fitting_MLE(frequency=frequency, 
                                psd=power_density, 
                                initial_paras=para_guess, 
                                nyquist=nyquist, 
                                gran_num=granulation, starid=starid, dirname=output_dir)

    main.plot_with_fit(frequency=frequency, 
                       psd=power_density, 
                       psd_smooth=psd_smooth, 
                       parameters=para_MLE, 
                       nyquist=nyquist, type='withgaussian', 
                       gran_num=granulation, starid=starid + '_MLE', dirname=output_dir)
    
    para_MCMC = main.fitting_MCMC(frequency=frequency, 
                                  psd=power_density,
                                  parameters_MLE=para_MLE, 
                                  nyquist=nyquist, 
                                  bound=0.5, nsteps=600, nwalkers=20,
                                  gran_num=granulation, starid=starid, dirname=output_dir)

    main.plot_with_fit(frequency=frequency, 
                    psd=power_density, 
                    psd_smooth=psd_smooth, 
                    parameters=para_MCMC[:, 0], 
                    nyquist=nyquist, type='withgaussian', 
                    gran_num=granulation, starid=starid + '_MCMC', dirname=output_dir)

    print('----- Power spectrum fitting completed -----')

    dnu_result, lower_limit, upper_limit = main.dnu(frequency=frequency, psd=power_density, 
                                                    numax=para_MLE[-3], parameters=para_MLE, 
                                                    nyquist=nyquist, gran_num=granulation, 
                                                    echelle_plot=1, echelletype='double', 
                                                    starid=starid, dirname=output_dir)

    print('----- Delta_nu calculation completed -----')
    
    # except:

    #     print('Error')
    #     continue
