
import matplotlib.pyplot as plt
import numpy as np
import os
from pymatgen.analysis.diffraction import xrd
from pymatgen.core import Structure

xs = np.linspace(10, 80, 4500)
calculator = xrd.XRDCalculator()

def _run(fileName, data, predList,loc):
    plt.title(fileName)
    data = data / max(data) * 100
    plt.plot(xs, data, label='data')
    for pred in predList:
        struc = Structure.from_file(get_cif_by_id(pred[0],loc))
        pattern = calculator.get_pattern(struc, two_theta_range=(10, 80))
        ys = np.zeros_like(xs)
        for x, y in zip(pattern.x, pattern.y):
            idx = np.argmin(np.abs(x - xs))
            ys[idx] = y
        plt.plot(xs, ys, linewidth=0.5, label='{}'.format(''.join(pred[2].split(' '))))

    plt.legend()
    plt.savefig('figs/{}.png'.format(fileName))
    plt.close()

def get_cif_by_id(codid,loc):
    return os.path.join(loc,'strucs','{}.cif').format(str(int(codid)))