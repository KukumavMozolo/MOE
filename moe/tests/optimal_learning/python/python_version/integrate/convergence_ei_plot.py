import numpy as np
import pylab as py

def load():
    location = '/home/kaw/Dokumente/Thesis/results/BADMJEI_runs_500_pre_1_iters_80_N_0.2.npy'
    location2 = '/home/kaw/Dokumente/Thesis/results/BAMGP_runs_500_pre_1_iters_80.npy'
    location3 = '/home/kaw/Dokumente/Thesis/results/MGP_runs_500_pre_1_iters_80.npy'
    location4 = '/home/kaw/Dokumente/Thesis/results/TSMGP_runs_500_pre_1_iters_80.npy'
    location5 = '/home/kaw/Dokumente/Thesis/results/BAMGP_runs_500_pre_1_iters_80_opta.npy'
    location6 = '/home/kaw/Dokumente/Thesis/results/BADMJEI_runs_500_pre_1_iters_80_N_0.1_opta.npy'
    location7 = '/home/kaw/Dokumente/Thesis/results/DMJEI_runs_500_pre_1_iters_80_N_0.2.npy'
    BADMJEI = np.load(location)
    BAMGP = np.load(location2)
    MGP = np.load(location3)
    TSMGP = np.load(location4)
    BAMGP_opt = np.load(location5)
    BADMJEI_op = np.load(location6)
    DMJEI = np.load(location7)
    convs = {'BADMJEI':BADMJEI}
    convs.update({'BAMGP': BAMGP})
    convs.update({'MGP': MGP})
    convs.update({'TSMGP': TSMGP})
    convs.update({'BAMGP_opt': BAMGP_opt})
    convs.update({'BADMJEI_op': BADMJEI_op})
    convs.update({'DMJEI': DMJEI})
    return convs

def add_plot(data, title, color, show_var = True):
    n,m = np.shape(data)
    if n == 500 and m ==80 or True:
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
    location = '/home/kaw/Dokumente/Thesis/results/BADMJEI_runs_500_pre_1_iters_80_N_0.2_ei.npy'
    location2 = '/home/kaw/Dokumente/Thesis/results/BAMGP_runs_500_pre_1_iters_80_ei.npy'
    location3 = '/home/kaw/Dokumente/Thesis/results/MGP_runs_500_pre_1_iters_80_ei.npy'
    location4 = '/home/kaw/Dokumente/Thesis/results/TSMGP_runs_500_pre_1_iters_80_ei.npy'
    location5 = '/home/kaw/Dokumente/Thesis/results/BAMGP_runs_500_pre_1_iters_80_opta_ei.npy'
    location6 = '/home/kaw/Dokumente/Thesis/results/BADMJEI_runs_500_pre_1_iters_80_N_0.1_opta_ei.npy'
    location7 = '/home/kaw/Dokumente/Thesis/results/DMJEI_runs_500_pre_1_iters_80_N_0.2_ei.npy'
    BADMJEI = np.load(location)
    BAMGP = np.load(location2)
    MGP = np.load(location3)
    TSMGP = np.load(location4)
    BAMGP_opta = np.load(location5)
    BADMJEI_opta = np.load(location6)
    DMJEI = np.load(location7)
    convs = {'BADMJEI':BADMJEI}
    convs.update({'BAMGP': BAMGP})
    convs.update({'MGP': MGP})
    convs.update({'TSMGP': TSMGP})
    convs.update({'BMGPA_opta': BAMGP_opta})
    convs.update({'BADMJEI_opta':BADMJEI_opta})
    convs.update({'DMJEI': DMJEI})
    return convs
def add_plot_ei(data, title, color, show_var = True):
    n,m = np.shape(data)
    if n == 500 and m ==80:
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
# py.rc('text', usetex=True)
# py.rc('font', family='serif')
plot_var = False
add_plot(convs['BADMJEI'], 'BADMJEI_N=5', 'red', plot_var)
add_plot(convs['BAMGP'], 'BAMGP', 'green', plot_var)
add_plot(convs['MGP'], 'MGP', 'blue', plot_var)
add_plot(convs['TSMGP'], 'TSMGP', 'black', plot_var)
add_plot(convs['BAMGP_opt'], 'BAMGP_op', 'purple', plot_var)
# add_plot(convs['BADMJEI_op'], 'BADMJEI_op', 'yellow', plot_var)
add_plot(convs['DMJEI'], 'DMJEI_N=5', 'mediumturquoise', plot_var)
py.legend()
py.xlabel('Iterations')
py.ylabel(r'log $|\Delta \frac{\pi}{10}|$')
py.title(r'$\sigma_n = 0.3, \sigma_f = 0.13, l_1=0.29, l_2 = 0.3$' )

py.subplot(1,2,2)
convs = load_ei()
add_plot_ei(convs['BADMJEI'], 'BADMJEI_N=5', 'red', plot_var)
add_plot_ei(convs['BAMGP'], 'BAMGP', 'green', plot_var)
add_plot_ei(convs['MGP'], 'MGP', 'blue', plot_var)
add_plot_ei(convs['TSMGP'], 'TSMGP', 'black', plot_var)
add_plot_ei(convs['BMGPA_opta'], 'BMGPA_op', 'purple', plot_var)
# add_plot_ei(convs['BADMJEI_opta'], 'BADMJEI_op', 'green', plot_var)
add_plot_ei(convs['DMJEI'], 'DMJEI_N=5', 'mediumturquoise', plot_var)
#py.legend(bbox_to_anchor=(0.62, 0.28))
py.xlabel('Iterations')
py.ylabel('log EI')
py.title(r'$\sigma_n = 0.3, \sigma_f = 0.13, l_1=0.29, l_2 = 0.3$' )

location = '/home/kaw/Dokumente/Thesis/plots/'
location2='/home/kaw/Dropbox/Thesis/plots/'
if plot_var:
    py.savefig(location + 'convergence_ei_plot_var.jpg')
    py.savefig(location2 + 'convergence_ei_plot_var.jpg')
else:
    py.savefig(location + 'convergence_ei_plot.jpg')
    py.savefig(location2 + 'convergence_ei_plot.jpg')
py.show()


