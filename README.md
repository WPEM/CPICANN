
## [WPEM Project](https://bin-cao.github.io/caobin/-wpem) 

<h1 align="center">
  <a href=""><img src="https://github.com/WPEM/CPICANN/assets/86995074/a40efe75-d5a9-4777-9d2a-cb4bed912d53" alt="CPICANN" width="250"></a>
  <br>
  <br>
</h1>

### Crystallographic Phase Identifier of Convolutional self-Attention Neural Network (CPICANN)

<img width="1289" alt="Screenshot 2024-02-24 at 08 42 24" src="https://github.com/WPEM/CPICANN/assets/86995074/eb3bf532-2281-49b7-b91e-bc7d53568c41">

### Introduction
This repo contains model and inference code for XRD phase identification by Deep Convolutional Self-Attention Neural Network. 

+ **Logo:** CPICANN Logo
+ **Paper:** CPICANN Paper
+ **src:** CPICANN Source Code
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


