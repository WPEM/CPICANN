<table>
  <tr>
    <td width="160" align="center" valign="top">
      <img src="https://github.com/user-attachments/assets/7e77bd5a-42d6-45db-b8e6-2c82cac81b9d" width="140" style="border-radius: 50%;"/>
    </td>
    <td valign="top">
      <b>For any inquiries or assistance, feel free to contact Mr. CAO Bin at:</b><br>
      📧 Email: <a href="mailto:bcao686@connect.hkust-gz.edu.cn">bcao686@connect.hkust-gz.edu.cn</a><br><br>
      Cao Bin is a PhD candidate at the <b>Hong Kong University of Science and Technology (Guangzhou)</b>, 
      under the supervision of Professor <a href="https://gbaaa.org.hk/en-us/article/67">Zhang Tong-Yi</a>. His research focuses on 
      <b>AI for science</b>, especially intelligent crystal-structure analysis and discovery. 
      Learn more about his work on his 
      <a href="https://www.caobin.asia/">homepage</a>.
    </td>
  </tr>
</table>

---

2024–2025, we achieved two key milestones in intelligent crystal phase identification. We open-sourced a high-fidelity PXRD simulation package, [**PySimXRD**](https://github.com/Bin-Cao/SimXRD), and released an intelligent identification platform, [**XQueryer**](https://github.com/Bin-Cao/XQueryer). Welcome to follow my latest work.

--
<h1 align="center">
  <a href="https://bin-cao.github.io/caobin/-wpem"><img src="https://github.com/Bin-Cao/WPEM/assets/86995074/3b05f104-364e-4cd2-9d21-f40b77e0ef10" alt="WPEM" width="250"></a>
  <br>
  <br>
</h1>

### Crystallographic Phase Identifier of Convolutional self-Attention Neural Network (CPICANN)

<img width="1289" alt="Screenshot 2024-02-24 at 08 42 24" src="https://github.com/WPEM/CPICANN/assets/86995074/eb3bf532-2281-49b7-b91e-bc7d53568c41">

### Introduction
This repo contains model and inference code for XRD phase identification by Deep Convolutional Self-Attention Neural Network. 

+ **Logo:** CPICANN Logo
+ **Paper:** CPICANN [Paper](https://journals.iucr.org/m/issues/2024/04/00/fc5077/fc5077.pdf)
+ **src:** CPICANN [Source Code](https://huggingface.co/AI4Cryst/CPICANN)
+ **Opendata**: [Data Sharing](https://huggingface.co/datasets/caobin/datasetCPICANN)
___
### Data Sharing:
You can access all the training and testing data via [**datasetCPICANN**](https://huggingface.co/datasets/caobin/datasetCPICANN) and the pretrained models via [**pretrainCPICANN**](https://huggingface.co/caobin/pretrainCPICANN). For further collaboration, please feel free to contact our research team.

### [SimXRD-4M](https://github.com/Bin-Cao/SimXRD)
**SimXRD** comprises 4,065,346 simulated powder X-ray diffraction patterns, representing 119,569 distinct crystal structures under 33 simulated conditions that mimic realworld variations. [**arxiv**](https://arxiv.org/pdf/2406.15469v1)
___
### Replication
If you wish to reproduce the results of our work or train the CPICANN model based on your own mission /data,  please refer to [instruction](https://github.com/WPEM/CPICANN/tree/main/src).

### Main Results


### Synthetic Single-phase Spectra
|      | whole test dataset | randomly sampled<br>(with elemental info) |
|------|--------------------|-------------------------------------------|
| CPICANN | 87.5%              | 99%                                       | 
| JADE | 38.7%              | 65%                                       | 

*Note: Results of JADE were obtained by using a customed crystal structure database which only contains the structures in directory /strucs.*

#### Synthetic Di-phase Spectra
<img width="373" alt="image" src="https://github.com/WPEM/CPICANN/assets/86995074/34b14780-0c1a-4169-8dd7-6b437f14df3f">

#### 100 experimental Spectra
<img width="425" alt="image" src="https://github.com/WPEM/CPICANN/assets/86995074/eba44550-a8ba-4340-ba06-daee7d394638">


### Installing / 安装
    pip install WPEMPhase 
    
### Checking / 查看
    pip show WPEMPhase 
    
### Updating / 更新
    pip install --upgrade WPEMPhase




### Template [CODE](https://github.com/WPEM/CPICANN/blob/main/src/inference%26case/CPICANNcode.ipynb) 

---
### Contact Information:
**Mr. Cao Bin**
Email: bcao686@connect.hkust-gz.edu.cn


### Acknowledgement:
If you utilize the data/code from this repo, please reference our paper.

### citation

``` javascript
@article{Zhang:fc5077,
author = "Zhang, Shouyang and Cao, Bin and Su, Tianhao and Wu, Yue and Feng, Zhenjie and Xiong, Jie and Zhang, Tong-Yi",
title = "{Crystallographic phase identifier of a convolutional self-attention neural network (CPICANN) on powder diffraction patterns}",
journal = "IUCrJ",
year = "2024",
volume = "11",
number = "4",
pages = "634--642",
month = "Jul",
doi = {10.1107/S2052252524005323},
url = {https://doi.org/10.1107/S2052252524005323},
abstract = {Spectroscopic data, particularly diffraction data, are essential for materials characterization due to their comprehensive crystallographic information. The current crystallographic phase identification, however, is very time consuming. To address this challenge, we have developed a real-time crystallographic phase identifier based on a convolutional self-attention neural network (CPICANN). Trained on 692{\hskip0.16667em}190 simulated powder X-ray diffraction (XRD) patterns from 23{\hskip0.16667em}073 distinct inorganic crystallographic information files, CPICANN demonstrates superior phase-identification power. Single-phase identification on simulated XRD patterns yields 98.5 and 87.5% accuracies with and without elemental information, respectively, outperforming {\it JADE} software (68.2 and 38.7%, respectively). Bi-phase identification on simulated XRD patterns achieves 84.2 and 51.5% accuracies, respectively. In experimental settings, CPICANN achieves an 80% identification accuracy, surpassing {\it JADE} software (61%). Integration of CPICANN into XRD refinement software will significantly advance the cutting-edge technology in XRD materials characterization.},
keywords = {computational modeling, structure prediction, X-ray diffraction, powder diffraction, phase identification, convolutional self-attention, autonomous characterization, neural networks, CPICANN},
}
``` 
