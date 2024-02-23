
<h1 align="center">
  <a href=""><img src="https://github.com/WPEM/CPICANN/assets/86995074/a40efe75-d5a9-4777-9d2a-cb4bed912d53" alt="CPICANN" width="250"></a>
  <br>
  <br>
</h1>

## Crystallographic Phase Identifier of Convolutional self-Attention Neural Network (CPICANN)
<img width="510" alt="image" src="https://github.com/WPEM/CPICANN/assets/86995074/be945b5c-8ff0-4a22-b0b2-7fe3132b4b7d">


## Introduction
This repo contains model and inference code for XRD phase identification by Deep Convolutional Self-Attention Neural Network. 

## Main Results


### Synthetic Single-phase Spectrum
|      | whole test dataset | randomly sampled<br>(with elemental info) |
|------|--------------------|-------------------------------------------|
| CPICANN | 87.5%              | 99%                                       | 
| JADE | 38.7%              | 65%                                       | 

*Note: Results of JADE were obtained by using a customed crystal structure database which only contains the structures in directory /strucs.*

### Synthetic Di-phase Spectrum
<img width="373" alt="image" src="https://github.com/WPEM/CPICANN/assets/86995074/34b14780-0c1a-4169-8dd7-6b437f14df3f">

### 100 experimental Spectrum
<img width="425" alt="image" src="https://github.com/WPEM/CPICANN/assets/86995074/eba44550-a8ba-4340-ba06-daee7d394638">


## Installing / 安装
    pip install WPEMPhase 
    
## Checking / 查看
    pip show WPEMPhase 
    
## Updating / 更新
    pip install --upgrade WPEMPhase




## Template [CODE](https://github.com/WPEM/CPICANN/blob/main/Template/CPICANNcode.ipynb) 
``` javascript
from WPEMPhase import CPICANN
CPICANN.PhaseIdentifier(FilePath='./testdata',Task='single-phase',)


Code introduction :


Signature:
CPICANN.PhaseIdentifier(
    FilePath,
    Task='single-phase',
    ElementsSystem='',
    ElementsContained='',
    ElementsExclude='',
    Device='cuda:0',
    CIFfiles=None,
    NNparam=None,
)
Docstring:
CPICANN : Crystallographic Phase Identifier of Convolutional self-Attention Neural Network

Contributors : Shouyang Zhang & Bin Cao
================================================================
    Please feel free to open issues in the Github :
    https://github.com/WPEM/CPICANN
    or 
    contact Mr.Bin Cao (bcao686@connect.hkust-gz.edu.cn)
    in case of any problems/comments/suggestions in using the code. 
==================================================================

:param FilePath 

:param Task, type=str, default='single-phase'
    if Task = 'single-phase', CPICANN executes a single phase identification task
    if Task = 'di-phase', CPICANN executes a dual phase identification task

:param ElementsSystem, type=str, default=''
    Specifies the elements to be included at least in the prediction, example: 'Fe'.

:param ElementsContained, type=str, default=''
    Specifies the elements to be included, with at least one of them in the prediction, example: 'O_C_S'.

:param ElementsExclude, type=str, default=''
    Specifies the elements to be excluded in the prediction, example: 'Fe_O'

:param Device, type=str, default='cuda:0',
    Which device to run the CPICANN, example: 'cuda:0', 'cpu'.

:param CIFfiles, type=str, default=None,
    at the first time execution, user need to assign the location of cifs' database, viz. strucs.zip file we provided, 

:param NNparam, type=str, default=None,
    at the first time execution, user need to assign the location of pretrained NN parameters, viz. pretrained.zip file we provided, 

examples:
from WPEMPhase import CPICANN
CPICANN.PhaseIdentifier(FilePath='./single-phase',Device='cpu')
File:      ~/miniconda3/lib/python3.9/site-packages/WPEMPhase/CPICANN.py
Type:      function
```

---
**About [WPEM](https://github.com/Bin-Cao/WPEM)** :

[WPEM](https://github.com/Bin-Cao/WPEM) specializes in elucidating intricate crystal structures and untangling heavily overlapping Bragg peaks in mixed X-rays and polycrystalline powder diffraction. Our endeavors have yielded noteworthy research outcomes, including the precise detection of subtle structural differences, such as the α phase and α' phase of Ti-Ni alloy, the differentiation of amorphous and crystalline contributions in Polybutene, the investigation of complex solid solution structures, and the computation of scattering information within organized processes. We are eager to engage in extensive collaborations and offer support in X-ray diffraction pattern refinement. For inquiries or assistance, please don't hesitate to contact us at bcao686@connect.hkust-gz.edu.cn (Dr. CAO Bin).

Our development journey commenced in 2020, driven by a commitment to patience and perfection in our work. Upon the publication of our final paper, we plan to freely share all our code promptly.
