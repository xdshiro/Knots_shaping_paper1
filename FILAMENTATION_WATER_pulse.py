import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from scipy.integrate import odeint, complex_ode, ode
from scipy.special import jv, iv
import plotly.graph_objects as go

# %% Modules
# checking spectra
show_field = True  # initial field
module_CheckingSpectrum = 1
module_2D = 1
# module_1D = 0
module_ADI = 0
module_Paraxial = 1
module_Intensity = 1  # 1-yes, 0-field abs

# %% Resolutions
xResolution = 51  # N_per
yResolution = 51
tResolution = 101
loopInnerResolution, loopOuterResolution = 301, 1  # M, Kmax
pulseTimeResolution = tResolution
zResolution = loopInnerResolution * loopOuterResolution
xStart, xFinish = 0, 1000 * 1e-6  # 10
yStart, yFinish = 0, 1000 * 1e-6  # 10
zStart, zFinish = 0, 0.11
tStart, tFinish = 0, 450 * 1e-15
tStart, tFinish = 0, 1550 * 1e-15

x0 = (xFinish - xStart) / 2
y0 = (yFinish - yStart) / 2
t0 = (tFinish - tStart) / 2

timeInZ = 0.1  # should be a little bit less than zFinish.

# %% Pulse parameters
rho0 = 200 * 1e-6
tp = 50 * 1e-15  # pulse duration
lambda0 = 0.517e-6
Pmax = 50e6  # 2.5e7
Pmax = 25e6  # 2.5e7
f = 1e5  # focus distance ??
lOAM = 1  # ????

# %% medium parameters
sigma_K8 = 2.4e-42 * 1e-2 ** (2 * 4)
sigma = 4e-18 * 1e-2 ** 2  #
rho_at = 7e22 * 1e-2 ** (-3)
a = 0
tau_c = 3e-15

# linear medium parameter

k2Dis = 5.6e-28 / 1e-2  # ps2/m  GVD
# k2Dis = -9.443607756116762e-22  # OAM non-diffraction
n0 = 1.332
# %% Nonlinear parameters
K = 4  # photons number
# %% temporal
C = 0  # chirp
n2 = 2.7e-16 * 1e-2 ** 2  # 3 * chi3 / (4 * eps0 * c * n0 ** 2)
# kDis = 0 * 0.02  # dispersion
q_e = -1.602176565e-19  # [C] electron charge
Ui = 7.1 * abs(q_e)
SigmaK = [0, 1, 2]


def Betta_func(K):
	Betta = [0, 0 * 2e-0, 2, 3, 2.4e-37 * 1e-2 ** (2 * K - 3), 5, 6, 7, 3.79347046850176e-121]
	return Betta[K]


# %% Constants
eps0 = 8.854187817e-12  # [F/m] - vacuum permittivity
cSOL = 2.99792458e8  # [m/s] speed of light in vacuum


def main():
	# %% Plotting parameters
	ticksFontSize = 18
	legendFontSize = 18
	xyLabelFontSize = 18
	# %% parameter recalculation
	k0 = 2 * np.pi / lambda0
	w0 = k0 * cSOL
	wD = 2 * n0 / (k2Dis * cSOL)
	# chi3 = n2 * 4 * eps0 * cSOL * n0 ** 2 / 3
	chi3_2 = 8 * n0 * n2 / 3
	epsNL = 3 * chi3_2 / 4
	Int = module_Intensity + 1
	Imax = 1
	if n2 == 0:
		Pcrit = 1e100
	else:
		Pcrit = 1.22 ** 2 * np.pi * lambda0 ** 2 / (32 * n0 * n2)
	print("P crit (MW): ", Pcrit * 1e-6)
	
	def LDF():
		return np.pi * rho0 ** 2 / lambda0
	
	def Lcollapse():
		temp1 = 0.367 * LDF()
		
		if n2 == 0:
			return 0
		else:
			temp2 = (np.sqrt(Pmax / Pcrit) - 0.852) ** 2 - 0.0219
			return temp1 / np.sqrt(temp2)
	
	print("Rayleigh length: ", LDF(), " Kerr Collapse length: ", Lcollapse())
	
	# %% Arrays creation
	xArray = np.linspace(xStart, xFinish, xResolution)
	yArray = np.linspace(yStart, yFinish, yResolution)
	zArray = np.linspace(zStart, zFinish, zResolution)
	tArray = np.linspace(tStart, tFinish, tResolution)
	# xtMesh = np.array(np.meshgrid(xArray, tArray, indexing='ij'))  # only ADI
	kxArray = np.linspace(-1. * np.pi * (xResolution - 2) / xFinish,
	                      1. * np.pi * (xResolution - 2) / xFinish, xResolution)
	kyArray = np.linspace(-1. * np.pi * (yResolution - 2) / yFinish,
	                      1. * np.pi * (yResolution - 2) / yFinish, yResolution)
	wArray = np.linspace(-1. * np.pi * (tResolution - 2) / tFinish, 1. * np.pi * (tResolution - 2) / tFinish,
	                     tResolution)
	xytMesh = np.array(np.meshgrid(xArray, yArray, tArray, indexing='ij'))
	KxywMesh = np.array(np.meshgrid(kxArray, kyArray, wArray, indexing='ij'))
	
	# initial fields
	def Field(x, y, t):
		return Imax * (np.exp(- (radius(x - x0, y - y0) ** 2) / rho0 ** 2 -
		                      1j * k0 * radius(x - x0, y - x0) ** 2 / (2 * f) -
		                      ((t - t0) / tp) ** 2)) * (
				(x - x0) / rho0 + 1j * np.sign(lOAM) * (y - y0) / rho0) ** np.abs(lOAM)
	
	# %% general functions
	def radius(x, y):
		return np.sqrt(x ** 2 + y ** 2)
	
	def split_step_old_time_Z(shape, loopInnerM=1, loopOuterKmax=1):
		def I(E):
			# page 44
			return np.abs(E) ** 2
		
		def plasma_density(E):
			plasmaDensity = np.zeros((xResolution, yResolution, tResolution))

			def Wofi(I):
				return sigma_K8 * I ** (K)
			
			def Wava(I):
				return sigma * I / Ui
			
			def Q_pd(I):
				return Wofi(I)
			
			def a_pd(I1, I2):
				tempValue = (tArray[1] - tArray[0]) * ((Wofi(I1) - Wava(I1)) + (Wofi(I2) - Wava(I2))) / 2
				
				return np.exp(-1 * tempValue)
			
			etta_pd = (tArray[1] - tArray[0]) * rho_at / 2
			
			for i in range(tResolution - 1):
				plasmaDensity[:, :, i + 1] = (a_pd(I(E[:, :, i]), I(E[:, :, i + 1])) *
				                              (plasmaDensity[:, :, i] + etta_pd * Q_pd(I(E[:, :, i])))
				                              + etta_pd * Q_pd(I(E[:, :, i + 1])))
			# - 0 * (tArray[1] - tArray[0]) * a * plasmaDensity[:, :, i] ** 2
			"""
											  + (tArray[1] - tArray[0]) *
			 (sigma_K8 * abs(E[:, :, i]) ** (2 * K) * (rho_at - plasmaDensity[:, :, i])
			  + sigma / Ui * abs(E[:, :, i]) ** 2 * plasmaDensity[:, :, i]
			  - a * plasmaDensity[:, :, i] ** 2))
			"""
			
			return plasmaDensity
		
		def Nonlinearity_spec(E):
			
			# print((w0 * n2 / cSOL * I(E)).max(), (sigma / 2 * (1) * plasmaDensity).max(),
			#      (sigma / 2 * (1j * w0 * tau_c) * plasmaDensity).max())
			# print((plasmaDensity).max())
			return dz * ((1j / (2 * eps0)) * ((w0 + KxywMesh[2]) / cSOL / n0) * eps0 * epsNL * I(E)
			             - Betta_func(K) / 2 * I(E) ** (K - 1) * (1 - plasmaDensity / rho_at)
			             - sigma / 2 * (1 + 1j * w0 * tau_c) * plasmaDensity)
		
		# return dz * 1j * w0 * n2 * I(E) / cSOL - dz * Betta_func(K) * I(E) ** (K - 1)  # & E
		
		E = shape(xytMesh[0], xytMesh[1], xytMesh[2])
		
		# print(E[int(xResolution / 2), int(yResolution / 2), int(tResolution / 2)])
		
		# works fine!
		def linear_step(field):
			temporaryField = fftshift(fftn(field))
			temporaryField = (temporaryField *
			                  np.exp(-1j * dz / (2 * k0 * n0) * KxywMesh[0] ** 2) *
			                  np.exp(-1j * dz / (2 * k0 * n0) * KxywMesh[1] ** 2) *
			                  np.exp(1j * dz * k2Dis / 2 * KxywMesh[2] ** 2))  # something here in /2
			return ifftn(ifftshift(temporaryField))
		
		def find_nearest_posi(array, value):
			array = np.asarray(array)
			idx = (np.abs(array - value)).argmin()
			return idx
		
		fieldReturn = np.zeros((xResolution, yResolution, 0), dtype=complex)
		
		Vg = cSOL / (n0 + 2 * n0 * w0 / wD)
		time = timeInZ / Vg
		print('time (ps): ', time * 1e12)
		
		# (tFinish / 2 + time - zFinish / cSOL / n0) * 1e15 < 0 or (t0 + time) > tFinish:
		zArrayWithTime = np.zeros(0)
		zCut = 1
		zArrayTemp = zArray
		temporalZCalc = 0
		newIndexTime = 1
		for k in range(loopOuterKmax):
			if module_CheckingSpectrum:
				Etest = E
				plt.plot(xArray, abs(Etest[:, int(yResolution / 2), int(tResolution / 2)]))
				plt.show()
				plt.close()
				plt.plot(yArray, abs(Etest[int(xResolution / 2), :, int(tResolution / 2)]))
				plt.show()
				plt.close()
				plt.plot(tArray, abs(Etest[int(xResolution / 2), int(yResolution / 2), :]))
				plt.show()
				plt.close()
				Etest = fftshift(fftn(E))
				plt.plot(kxArray, abs(Etest[:, int(yResolution / 2), int(tResolution / 2)]))
				plt.show()
				plt.close()
				plt.plot(kyArray, abs(Etest[int(xResolution / 2), :, int(tResolution / 2)]))
				plt.show()
				plt.close()
				plt.plot(wArray, abs(Etest[int(xResolution / 1.5), int(yResolution / 1.5), :]))
				plt.show()
				plt.close()
			for m in range(1, loopInnerM):
				zInd = (k) * loopInnerM + m
				dz = zArray[zInd] - zArray[zInd - 1]
				
				if zCut:
					plasmaDensity = plasma_density(E)
					# print(np.abs(E).max())
					
					E = linear_step(E)
					# print(plasmaDensity.max())
					E = E * np.exp(Nonlinearity_spec(E))
					print(zInd)
					timeTau = t0 + time - zArray[zInd] / Vg
					if timeTau <= tFinish:
						zCut = 0
						zArrayWithResolutionTime = (np.linspace(0, tFinish * Vg, pulseTimeResolution)
						                            + time * Vg - t0 * Vg)
						# print(zArrayWithResolutionTime)
						timeTau = t0 + time - zArrayWithResolutionTime[newIndexTime] / Vg  #
						timeTemp = find_nearest_posi(tArray, timeTau)
						zArrayTemp = np.zeros((1))
						
						zArrayTemp = np.append(zArrayTemp, zArrayWithResolutionTime[newIndexTime] -
						                       (tArray[timeTemp] - timeTau) * Vg)
						if round(zArrayTemp[newIndexTime], 9) != round(zArrayTemp[newIndexTime - 1], 9):
							fieldReturn = np.concatenate([fieldReturn[:, :, :], E[:, :, None, timeTemp]], axis=2)
							zArrayWithTime = np.append(zArrayWithTime, zArrayTemp[newIndexTime])
						newIndexTime += 1
				else:
					while newIndexTime < pulseTimeResolution:
						timeTau = t0 + time - zArrayWithResolutionTime[newIndexTime] / Vg  #
						# print(timeTau)
						timeTemp = find_nearest_posi(tArray, timeTau)
						zArrayTemp = np.append(zArrayTemp, zArrayWithResolutionTime[newIndexTime] -
						                       (tArray[timeTemp] - timeTau) * Vg)
						dz = zArrayTemp[newIndexTime] - zArrayTemp[newIndexTime - 1]
						
						if round(zArrayTemp[newIndexTime], 9) != round(zArrayTemp[newIndexTime - 1], 9):
							fieldReturn = np.concatenate([fieldReturn[:, :, :], E[:, :, None, timeTemp]], axis=2)
							zArrayWithTime = np.append(zArrayWithTime, zArrayTemp[newIndexTime])
						else:
							print(zArrayTemp)
							temporalZCalc += 1
						newIndexTime += 1
					
					break
			if module_CheckingSpectrum:
				Etest = fftshift(fftn(E))
				plt.plot(kxArray, abs(Etest[:, int(yResolution / 2), int(tResolution / 2)]))
				plt.show()
				plt.close()
				plt.plot(kyArray, abs(Etest[int(xResolution / 2), :, int(tResolution / 2)]))
				plt.show()
				plt.close()
				plt.plot(wArray, abs(Etest[int(xResolution / 1.5), int(yResolution / 1.5), :]))
				plt.show()
				plt.close()
		print('ZT Points lost: ', temporalZCalc)
		print('z boundaries: ', zArrayWithTime[0], zArrayWithTime[-1])
		# plt.plot(tArray, abs(plasmaDensity[int(xResolution/2), int(xResolution/2), :]))
		# plt.show()
		# exit()
		
		return fieldReturn, zArrayWithTime, E
	
	def plot_2D(E, x, y, xname='', yname='', map='jet', vmin=0.13, vmax=1.14):
		if vmin == 0.13 and vmax == 1.14:
			vmin = E.min()
			vmax = E.max()
		
		image = plt.imshow(E,
		                   interpolation='bilinear', cmap=map,
		                   origin='lower', aspect='auto',  # aspect ration of the axes
		                   extent=[y[0], y[-1], x[0], x[-1]],
		                   vmin=vmin, vmax=vmax, label='sdfsd')
		cbr = plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
		cbr.ax.tick_params(labelsize=ticksFontSize)
		plt.xticks(fontsize=ticksFontSize)
		plt.yticks(fontsize=ticksFontSize)
		ax.set_xlabel(xname, fontsize=xyLabelFontSize)
		ax.set_ylabel(yname, fontsize=xyLabelFontSize)
	
	if module_2D:
		E = Field(xytMesh[0], xytMesh[1], xytMesh[2])[:, int(yResolution / 2), :]
		fig, ax = plt.subplots(figsize=(8, 7))
		if show_field:
			plot_2D(np.abs(E) ** Int, xArray * 1e3, tArray, 't (s)',
			        'x (mm)',
			        map='viridis')
			plt.show()
			E = Field(xytMesh[0], xytMesh[1], xytMesh[2])[:, :, int(tResolution / 2)]
			plot_2D(np.abs(E) ** Int, yArray * 1e3, xArray * 1e3, 'y (mm)',
			        'x (mm)',
			        map='viridis')
			plt.show()
		Sq = np.sum(np.abs(E) ** 2) * (xArray[1] - xArray[0]) * (yArray[1] - yArray[0])
		# P_cr = P_critical_initialization(wavelength)  # дж / с
		Imax = np.sqrt(Pmax / Sq)
		print('max intensity: ', Imax)
		
		"""fieldAdiZ = ADI_2D1_nonlinear_Z(Field_1_2D(xtMesh[0], xtMesh[1]),
									 loopInnerResolution, loopOuterResolution)"""
		fieldOLD, zArrayWithTime, fieldXYT = split_step_old_time_Z(Field, loopInnerResolution, loopOuterResolution)
		print(np.shape(fieldXYT))
		plot_2D(np.abs(fieldOLD[:, int(yResolution / 2), :]) ** Int, xArray * 1e3, zArrayWithTime * 1e3, 'z (mm)',
		        'x (mm)',
		        map='viridis')
		# plt.xlim(2.25 - 1, 2.25 + 1)
		# plt.ylim(2.25 - 1, 2.25 + 1)
		plt.title(f'z={round(timeInZ * 1e3, 0)}mm', fontweight="bold",
		          fontsize=26)
		plt.show()
		plt.close()
		# np.save('filam_fs_xz.npy', fieldOLD)
		plt.show()
		plot_2D(np.abs(fieldOLD[:, :, np.shape(fieldOLD)[2] // 2]) ** Int, yArray * 1e3, xArray * 1e3, 'y (mm)',
		        'x (mm)',
		        map='viridis')
		plt.show()
		plot_2D(np.abs(fieldXYT[:, np.shape(fieldOLD)[1] // 2, :]) ** Int, xArray * 1e3, tArray, 't (s)',
		        'x (mm)',
		        map='viridis')
		plt.show()
		exit()
	
	if 0:
		E = Field_STOV_simple(xytMesh[0], xytMesh[1], xytMesh[2])[:, :, int(tResolution / 2)]
		if module_initial:
			fig, ax = plt.subplots(figsize=(8, 7))
			plt.plot(tArray, np.abs(Field_STOV_simple(xytMesh[0], xytMesh[1], xytMesh[2])
			                        [int(xResolution / 2), int(yResolution / 2), :]) ** Int)
			# plt.xlim(t0-tFWHM/2,t0+tFWHM/2)
			plt.show()
			fig, ax = plt.subplots(figsize=(8, 7))
			plt.plot(xArray, np.abs(Field_STOV_simple(xytMesh[0], xytMesh[1], xytMesh[2])
			                        [:, int(yResolution / 2), int(tResolution / 2)]) ** Int)
			# plt.xlim(x0 - rFWHM / 2, x0 + rFWHM/2)
			plt.show()
		Sq = np.sum(E ** 2) * (xArray[1] - xArray[0]) * (yArray[1] - yArray[0])
		# P_cr = P_critical_initialization(wavelength)  # дж / с
		Imax = np.sqrt(Pmax / Sq)
		
		if module_Paraxial:
			fieldOLD = split_step_old_time(Field_STOV_simple, loopInnerResolution, loopOuterResolution)
		if module_NonRapaxial:
			fieldUPPE = UPPE_time(Field_STOV_simple, loopInnerResolution, loopOuterResolution)
		# 2D
		if module_2D:
			if module_Paraxial:
				fig, ax = plt.subplots(figsize=(8, 7))
				image = plt.imshow(np.abs(fieldOLD[:, :, int(tResolution / 2)]) ** Int,
				                   interpolation='bilinear', cmap='magma',
				                   origin='lower', aspect='auto',  # aspect ration of the axes
				                   extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
				plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
				ax.set_xlabel('y', fontsize=xyLabelFontSize)
				ax.set_ylabel('x', fontsize=xyLabelFontSize)
				plt.show()
			if module_NonRapaxial:
				fig, ax = plt.subplots(figsize=(8, 7))
				image = plt.imshow(np.abs(fieldUPPE[:, :, int(tResolution / 2)]) ** Int,
				                   interpolation='bilinear', cmap='magma',
				                   origin='lower', aspect='auto',  # aspect ration of the axes
				                   extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
				plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
				ax.set_xlabel('y', fontsize=xyLabelFontSize)
				ax.set_ylabel('x', fontsize=xyLabelFontSize)
				plt.show()
				plt.close()
		# 1D
		if module_1D:
			fig, ax = plt.subplots(figsize=(8, 7))
			# ax.plot(rArray, np.abs(Eps(rArray, z=zFinish)), color='lime', ls='-', label='Analytical', lw=4)
			
			# ax.plot(tArray, np.abs(Eps_dispersion(tArray, z=zFinish)), ls='-', label='Analytical', color='black', lw=10)
			"""fieldAdi = ADI_2D1_nonlinear(Eps_initiation_with_time(xtMesh[0], xtMesh[1]), loopInnerResolution,
										 loopOuterResolution)"""
			# ax.plot(tArray, np.abs(fieldAdi)[0, :], ls='-', label='ADI Crank Nicolson', color='red', lw=10)
			"""fieldAdiNonlinear = ADI_2D1_nonlinear(Eps_initiation_with_time(rtMesh[0], rtMesh[1]), loopInnerResolution,
										   loopOuterResolution)
			plot_1D(tArray, np.abs(fieldAdiNonlinear)[0, :]
					, label='Crank Nicolson Dispersion', xname=r'$t$', yname='', ls='-', color='red', lw=6)"""
			# fieldOLD = split_step_old_time(Field_1, loopInnerResolution, loopOuterResolution)
			if module_Paraxial:
				plot_1D(tArray, np.abs(fieldOLD[int(xResolution / 2), int(yResolution / 2), :] ** Int)
				        , label='Paraxial', xname=r'$t$', yname='', color='lime', ls='-', lw=6)
			# fieldUPPE = UPPE_time(Field_1, loopInnerResolution, loopOuterResolution)
			if module_NonRapaxial:
				plot_1D(tArray, np.abs(fieldUPPE[int(xResolution / 2), int(yResolution / 2), :] ** Int)
				        , label='UPPE (non-paraxial)', xname=r'$t$', yname='', color='blue', ls='-', lw=2)
			plt.show()
			# plt.xlim(15,30.5)
			# plt.show()
			# fig, ax = plt.subplots(figsize=(8, 7))
			# plt.plot(xArray, np.abs(fieldAdi[:, int(tResolution/2)]),
			#         color='red', ls='--', lw=6)
			# E(x)
			if 1:
				fig, ax = plt.subplots(figsize=(8, 7))
				if module_Paraxial:
					plt.plot(xArray, np.abs(fieldOLD[:, int(yResolution / 2), int(tResolution / 2)]) ** Int,
					         label='Paraxial', color='lime', ls='--', lw=6)
					plt.text(0, 0, f'z={zFinish * 1e3}mm', color='black', fontsize=26)
				if module_NonRapaxial:
					plot_1D(xArray, np.abs(fieldUPPE[:, int(yResolution / 2), int(tResolution / 2)]) ** Int,
					        label='UPPE (non-paraxial)', xname=r'$x$', yname='', color='blue', ls='--', lw=2)
				# plt.xlim(xStart, xFinish)
				plt.show()
		# phase
		if module_3D:
			# E = Field_1(xytMesh[0], xytMesh[1], xytMesh[2])[:, :, int(tResolution / 2)]
			# field3D = PSIl_N(xytMesh[0], xytMesh[1], 0, xytMesh[2])Field_STOV_simple
			# field3D = Field_STOV_simple(xytMesh[0], xytMesh[1], xytMesh[2])
			"""print(np.shape(field3D))
					fig, ax = plt.subplots(figsize=(8, 7))
					image = plt.imshow(np.abs(field3D[:, :, int(tResolution / 2)]) ** Int,
									   interpolation='bilinear', cmap='magma',
									   origin='lower', aspect='auto',  # aspect ration of the axes
									   extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
					plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
					ax.set_xlabel('y', fontsize=xyLabelFontSize)
					ax.set_ylabel('x', fontsize=xyLabelFontSize)
					plt.show()
					exit()
					print(field3D.max())"""
			plot_3D(fieldOLD)
		if module_Phase:
			if module_Paraxial:
				fig, ax = plt.subplots(figsize=(8, 7))
				image = plt.imshow(np.angle(fieldOLD[:, int(yResolution / 2), :]),
				                   interpolation='bilinear', cmap='jet',
				                   origin='lower', aspect='auto',  # aspect ration of the axes
				                   extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
				plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
				ax.set_xlabel('t', fontsize=xyLabelFontSize)
				ax.set_ylabel('x', fontsize=xyLabelFontSize)
				plt.show()
			if module_NonRapaxial:
				fig, ax = plt.subplots(figsize=(8, 7))
				image = plt.imshow(np.angle(fieldUPPE[:, int(yResolution / 2), :]),
				                   interpolation='bilinear', cmap='jet',
				                   origin='lower', aspect='auto',  # aspect ration of the axes
				                   extent=[tArray[0], tArray[-1], xArray[0], xArray[-1]])
				plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
				ax.set_xlabel('t', fontsize=xyLabelFontSize)
				ax.set_ylabel('x', fontsize=xyLabelFontSize)
				plt.show()
				plt.close()


if __name__ == '__main__':
	main()
