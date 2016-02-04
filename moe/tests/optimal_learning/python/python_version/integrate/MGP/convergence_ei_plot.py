import numpy as np
import pylab as py
from os.path import expanduser

def load():
    home = expanduser("~")
    location =  home +'/Documents/Thesis/results/BAMGP2_runs_500_pre_1_iters_150_opta.npy'
    location4 = home +'/Documents/Thesis/results/BAMGP_Uni_40_runs_500_pre_1_iters_150_opta.npy'
    location2 = home +'/Documents/Thesis/results/TSMGP_runs_500_pre_1_iters_150.npy'
    location3 = home +'/Documents/Thesis/results/MGP2_Uni_40_runs_500_pre_1_iters_150.npy'
    location5 = home +'/Documents/Thesis/results/BAMGP_greed_20_runs_500_pre_1_iters_150_opta.npy'
    BAMGP = np.load(location)
    TSMGP = np.load(location2)
    MGP = np.load(location3)
    BAMGP_Uni = np.load(location4)
    BAMGP_g = np.load(location5)
    convs = {'BAMGP':BAMGP}
    convs.update({'TSMGP' :TSMGP})
    convs.update({'MGP': MGP})
    convs.update({'BAMGP_Uni':BAMGP_Uni})
    convs.update({'BAMGP_g': BAMGP_g})
    return convs

def add_plot(data, title, color, show_var = True):
    n,m = np.shape(data)
    if n == 500 and m ==150 or True:
        abs_diffs = np.abs(data - np.pi/10.0)
        mean_abs_diffs = np.mean(abs_diffs, axis=0)
        logg_mean_abs_diffs = np.log(mean_abs_diffs)
        var = np.var(abs_diffs, axis=0)
        std = np.sqrt(var)
        if show_var:
            py.errorbar(range(m), logg_mean_abs_diffs, yerr=2*std, c=color, label=title)
        else:
            py.errorbar(range(m), logg_mean_abs_diffs,  c=color, label=title)
    else:
        raise ValueError('n, m are wrong: n = ' + str(n) + ' ,m = '+str(m))

def load_ei():
    home = expanduser("~")
    location = home +'/Documents/Thesis/results/BAMGP2_runs_500_pre_1_iters_150_opta_ei.npy'
    location2 = home +'/Documents/Thesis/results/TSMGP_runs_500_pre_1_iters_150_ei.npy'
    location3 = home +'/Documents/Thesis/results/MGP2_Uni_40_runs_500_pre_1_iters_150_ei.npy'
    location4 = home +'/Documents/Thesis/results/BAMGP_Uni_40_runs_500_pre_1_iters_150_opta_ei.npy'
    location5 = home +'/Documents/Thesis/results/BAMGP_greed_20_runs_500_pre_1_iters_150_opta_ei.npy'
    BAMGP = np.load(location)
    TSMGP = np.load(location2)
    MGP = np.load(location3)
    BAMGP_Uni = np.load(location4)
    BAMGP_g = np.load(location5)
    convs = {'BAMGP':BAMGP}
    convs.update({'TSMGP': TSMGP})
    convs.update({'MGP': MGP})
    convs.update({'BAMGP_Uni':BAMGP_Uni})
    convs.update({'BAMGP_g':BAMGP_g})
    return convs
def add_plot_ei(data, title, color, show_var = True):
    n,m = np.shape(data)
    if n == 500 and m ==150:
        mean = np.mean(data, axis=0)
        log_mean = np.log(mean)
        var = np.var(data, axis=0)
        std = np.sqrt(var)
        if show_var:
            py.errorbar(range(m), log_mean, yerr=2*std, c=color, label=title)
        else:
            py.errorbar(range(m), log_mean,  c=color, label=title)
    else:
        raise ValueError('n, m are wrong: n = ' + str(n) + ' ,m = '+str(m))




convs = load()
py.figure(figsize=(20,12))
py.subplot(1,2,1)

plot_var = False
add_plot(convs['BAMGP'], 'BAMGP', 'green', plot_var)
add_plot(convs['TSMGP'], 'TSMGP', 'black', plot_var)
add_plot(convs['MGP'], 'MGP', 'blue', plot_var)
add_plot(convs['BAMGP_Uni'], 'BAMGP_Uni', 'red', plot_var)
add_plot(convs['BAMGP_g'], 'BAMGP_g', 'turquoise', plot_var)

py.legend()
py.xlabel('Iterations')
py.ylabel(r'log $|\Delta \frac{\pi}{10}|$')
py.title(r'$\sigma_n = 0.3, \sigma_f = 0.33, l_1=0.24, l_2 = 0.25$' )

py.subplot(1,2,2)
convs = load_ei()
add_plot_ei(convs['BAMGP'], 'BAMGP', 'green', plot_var)
add_plot_ei(convs['TSMGP'], 'TSMGP', 'black', plot_var)
add_plot_ei(convs['MGP'], 'MGP', 'blue', plot_var)
add_plot_ei(convs['BAMGP_Uni'], 'BAMGP_Uni', 'red', plot_var)
add_plot_ei(convs['BAMGP_g'], 'BAMGP_g', 'turquoise', plot_var)
#py.legend(bbox_to_anchor=(0.62, 0.28))
py.xlabel('Iterations')
py.ylabel('log EI')
py.title(r'$\sigma_n = 0.3, \sigma_f = 0.33, l_1=0.24, l_2 = 0.25$' )

location = '//home/maxweule/Documents/Thesis/plots'
if plot_var:
    py.savefig(location + 'convergence_ei_plot_var.jpg')
else:
    py.savefig(location + 'convergence_ei_plot.jpg')
py.show()


