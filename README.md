# DeepPhaseIDentification

## Introduction
This repo contains model and inference code for XRD phase identification by Deep Convolutional Self-Attention Neural Network.

## Main Results


### Synthetic Single-phase Spectrum
|      | whole test dataset | randomly sampled<br>(with elemental info) | sampled by crystal system<br>(with elemental info) |
|------|--------------------|-------------------------------------------|----------------------------------------------------|
| DPID | 87.5%              | 99%                                       | 99%                                                |
| JADE | 38.7%              | 65%                                       | 53%                                                |
*Note: Results of JADE were obtained by using a customed crystal structure database which only contains the structures in directory /strucs.*

### Experimental Single-phase Spectrum
| Formula | Space Group | Crystal System | JADE            | DPID            |
|---------|-------------|----------------|-----------------|-----------------|
| Al2O3   | R-3c        | Hexagonal      | F               | T               | 
| CdS     | P6_3mc      | Hexagonal      | T               | T               |
| Mn2O3   | I2_13       | Cubic          | F               | F               |
| MnS     | Fm-3m       | Cubic          | T               | T               |
| NiO2H2  | P-3m1       | Hexagonal      | T               | T               |
| PbSO4   | Pnma        | Orthorhombic   | F               | T               |
| PbSe    | Fm-3m       | Cubic          | T               | T               |
| RuO2    | P4_2/mnm    | Tetragonal     | T               | T               |
| Sn      | I4_1/amd    | Tetragonal     | F               | T               |
| Ti(BCC) | Im-3m       | Cubic          | F               | F               |
| WO3     | P2_1/c      | Monoclinic     | T               | F               |
| ZnS     | F-43m       | Cubic          | F               | F               |
| Zr      | P6_3/mmc    | Hexagonal      | T               | T               |
|         |             | **Accuracy**   | **7/13(53.8%)** | **9/13(69.2%)** |
*Note: T for true and F for false*   
*Note: Results of JADE were obtained by using a customed crystal structure database which only contains the structures in directory /strucs.*

### Synthetic Di-phase Spectrum
<table><tbody>
    <th colspan="2" align="center">di-phase prediction results</th>
    <th align="center">Random mixed spectra</th>
    <th align="center">Fe corrosion</th>
    <tr>
        <td align="center">Contains at least<br>one phase</td>
        <td align="center">Top2</td>
        <td align="center">95.0%</td>
        <td align="center">98.8%</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">Contains<br>both phase</td>
        <td align="center">Top2</td>
        <td align="center">42.8%</td>
        <td align="center">77.8%</td>
    </tr>
    <tr>
        <td align="center">Top3</td>
        <td align="center">57.6%</td>
        <td align="center">82.6%</td>
    </tr>
    <tr>
        <td align="center">Top3</td>
        <td align="center">70.6%</td>
        <td align="center">96.4%</td>
    </tr>
</tbody></table>  

*Note: Results of Fe corrosion were obtained with random mixture of 20 selected Fe corrosion meterials, more details at [TBD]*

## Usage
### Preparation
Install pytorch and pymatgen(for inference results visualization).  
The code has been tested on CentOS 7.4 with Python 3.7.4, PyTorch 1.10.0, CUDA 10.2.

### Single-phase Inference
To perform single-phase inference, place your data at ```/data/single-phase``` and run ```inference.py```.  
You can specify data directory by changing the parameter ```data_path```.
```angular2html
python inference.py \
    -inf_mode single-phase \
    -data_path data/single-phase \ 
```

### Di-phase Inference
To perform di-phase inference, place your data at ```/data/di-phase``` and run ```inference.py```.  
You can specify data directory by changing the parameter ```data_path```.
```angular2html
python inference.py \
    -inf_mode di-phase \
    -data_path data/di-phase \ 
    -include_elements_must Fe \
    -include_elements_atLeastOne O_H_S
```