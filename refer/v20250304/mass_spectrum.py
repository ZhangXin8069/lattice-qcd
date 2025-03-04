import numpy as np
import matplotlib.pyplot as plt
Jp_pion=0
mass_pion=0.298
Jp_proton=1/2
mass_proton=1.134
Jp_sigma=1/2
mass_sigma=1.259
Jp_xi=1/2
mass_xi=1.306
xshft = 0.3
fig, ax = plt.subplots(1,1, figsize=(10, 7*0.5))
ax.set_ylim([0,3])
fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
plt.scatter(Jp_pion,mass_pion,Jp_proton,mass_proton)
plt.legend()
ax.set_xlabel('J.p')
ax.set_ylabel('$m_{\mathrm{eff}}$ [GeV]')
fig.savefig("./mass_spectrum.png")