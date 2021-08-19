#!/usr/bin/env python
import h5py

from astropy.coordinates import SkyCoord
from astropy import constants as const, units as u
import bilby
from gwpy.timeseries import TimeSeries
import numpy as np


outdir = '/Users/rickwilde/Desktop/summer_research/git'
# outdir = '/scrah/users/deep1018/GW170817-dynesty/inject-lambda-0-sample-z-flatz-prior-run-2'
label = 'inject-lambda-0-sample-z-flat-h0-prior-O5'
bilby.core.utils.setup_logger(outdir=outdir, label=label,
                              log_level='info')
logger = bilby.core.utils.logger

roll_off = 0.2  # Roll off duration of tukey window in seconds

# Injection parameters
# Inject the following signal and sample on the redshift
chirp_mass =  1.18
mass_ratio = 0.869
a_1 = 0.
a_2 = 0.
tilt_1 = 0.
tilt_2 = 0.
phi_12 = 0.
luminosity_distance = 200
theta_jn = 146./180. * np.pi
phi_jl = 0.
psi=2.659
phase=1.3

# These are two of the really important variables 
ra = RA_NGC_4993
dec = DEC_NGC_4993

trigger_time = 1187008882.44
sampling_frequency = 4096
duration = 128
start_time = trigger_time - duration

injection_parameters = dict(
    chirp_mass=chirp_mass, mass_ratio=mass_ratio, a_1=a_1, a_2=a_2,
    tilt_1=tilt_1, tilt_2=tilt_2, theta_jn=theta_jn,
    luminosity_distance=luminosity_distance, phi_jl=phi_jl,
    psi=psi, phase=phase, geocent_time=trigger_time, phi_12=phi_12,
    ra=ra, dec=dec
)

waveform_arguments = dict(waveform_approximant='TaylorF2',
                          reference_frequency=20.)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=my_conversion_function,
    waveform_arguments=waveform_arguments
)

# load PSD files for O5
psd_filenames = {
    'H1': '/home/deep1018/bilby-test-run/data/O5_PSDS/AplusDesign.txt',
    'L1': '/home/deep1018/bilby-test-run/data/O5_PSDS/AplusDesign.txt',
    'V1': '/home/deep1018/bilby-test-run/data/O5_PSDS/avirgo_O5high_NEW.txt'
}

ifo_list = bilby.gw.detector.InterferometerList([])
for det in ["H1", "L1", "V1"]:
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    freq, asd = np.loadtxt(psd_filenames[det], unpack=True)
    psd = asd**2
    logger.info(f"Setting PSD using GW170817 PSD for {det}")
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=freq, psd_array=psd
    )
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=start_time
    )
    ifo_list.append(ifo)

ifo_list.inject_signal(
    parameters=injection_parameters,
    waveform_generator=waveform_generator
)

logger.info("Finished Injecting signal")
logger.info("Saving IFO data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)

# create a GW170817 prior; sample in chirp_mass and mass_ratio
prior_dictionary = dict(
    chirp_mass=bilby.gw.prior.Uniform(name='chirp_mass', minimum=1.17, maximum=1.19),
    mass_ratio=bilby.gw.prior.Uniform(name='mass_ratio', minimum=0.65, maximum=1),
    mass_1=bilby.gw.prior.Constraint(name='mass_1', minimum=1.1, maximum=1.9),
    mass_2=bilby.gw.prior.Constraint(name='mass_2', minimum=1.1, maximum=1.9),
    a_1=bilby.gw.prior.Uniform(name='a_1', minimum=0, maximum=0.05,
                               latex_label='$a_1$', unit=None, boundary=None),
    a_2=bilby.gw.prior.Uniform(name='a_2', minimum=0, maximum=0.05,
                               latex_label='$a_2$', unit=None, boundary=None),
    tilt_1=bilby.prior.Sine(name='tilt_1', latex_label='$\\theta_1$', unit=None),
    tilt_2=bilby.prior.Sine(name='tilt_2', latex_label='$\\theta_2$', unit=None),
    phi_12=bilby.gw.prior.Uniform(name='phi_12', minimum=0, maximum=2 * np.pi,
                                  boundary='periodic', latex_label='$\\Delta\\phi$', unit=None),
    phi_jl=bilby.gw.prior.Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi,
                                  boundary='periodic', latex_label='$\\phi_{JL}$', unit=None),
    luminosity_distance=bilby.gw.prior.UniformComovingVolume(name='luminosity_distance',
                                                             minimum=10, maximum=500, latex_label='$d_L$',
                                                             unit='Mpc', boundary=None),
    dec=bilby.prior.Cosine(name='dec', latex_label='$\\mathrm{DEC}$',
                           unit=None, minimum=-np.pi / 2, maximum=np.pi / 2, boundary=None),
    ra=bilby.gw.prior.Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic',
                              latex_label='$\\mathrm{RA}$', unit=None),
    theta_jn=bilby.prior.Sine(name='theta_jn', latex_label='$\\theta_{JN}$',
                              unit=None, minimum=0, maximum=np.pi, boundary=None),
    psi=bilby.gw.prior.Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic',
                               latex_label='$\\psi$', unit=None)
)

priors = bilby.gw.prior.BBHPriorDict(dictionary=prior_dictionary)


# set a small margin on time of arrival 
priors['geocent_time'] = bilby.core.prior.DeltaFunction(
    peak=trigger_time
)

# Initialise the likelihood
# Ricky change: set time_marginalization to True
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifo_list, waveform_generator=waveform_generator,
    time_marginalization=True, phase_marginalization=True,
    distance_marginalization=False, priors=priors
)

logger.info("Calling dynesty...")

# dynesty options
# ricky change: you should consider what you want to do to the npool variable
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', dlogz=0.1,
    walks=100, check_point_delta_t=20000, npool=24, outdir=outdir, label=label,
    nlive=1000, n_effective=1000, injection_parameters=injection_parameters
)
