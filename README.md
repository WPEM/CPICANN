
<h1 align="center">
  <a href=""><img src="https://github.com/WPEM/CPICANN/assets/86995074/a40efe75-d5a9-4777-9d2a-cb4bed912d53" alt="CPICANN" width="250"></a>
  <br>
  <br>
</h1>

## Introduction
This repo contains model and inference code for XRD phase identification by Deep Convolutional Self-Attention Neural Network.

## Main Results


### Synthetic Single-phase Spectrum
|      | whole test dataset | randomly sampled<br>(with elemental info) | sampled by crystal system<br>(with elemental info) |
|------|--------------------|-------------------------------------------|----------------------------------------------------|
| DPID | 87.5%              | 99%                                       | 99%                                                |
| JADE | 38.7%              | 65%                                       | 53%                                                |  

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


## Installing / 安装
    pip install WPEMPhase 
    
## Checking / 查看
    pip show WPEMPhase 
    
## Updating / 更新
    pip install --upgrade WPEMPhase




## [Template](https://github.com/WPEM/CPICANN/tree/main/Template) 
``` javascript
from WPEMPhase import CPICANN
CPICANN.PhaseIdentifier(FilePath='./testdata',Task='single-phase',)
```

---
![WechatIMG954](https://github.com/Bin-Cao/WPEM/assets/86995074/65b44e3f-257b-4ea7-8b54-174a1359449f)


---
**About WPEM** :

WPEM specializes in elucidating intricate crystal structures and untangling heavily overlapping Bragg peaks in mixed X-rays and polycrystalline powder diffraction. Our endeavors have yielded noteworthy research outcomes, including the precise detection of subtle structural differences, such as the α phase and α' phase of Ti-Ni alloy, the differentiation of amorphous and crystalline contributions in Polybutene, the investigation of complex solid solution structures, and the computation of scattering information within organized processes. We are eager to engage in extensive collaborations and offer support in X-ray diffraction pattern refinement. For inquiries or assistance, please don't hesitate to contact us at bcao686@connect.hkust-gz.edu.cn (Dr. CAO Bin).

Our development journey commenced in 2020, driven by a commitment to patience and perfection in our work. Upon the publication of our final paper, we plan to freely share all our code promptly.
