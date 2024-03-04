# E2EPep
Protein-peptide binding residue prediction based on protein language models and feature fusion method

Accurate identifications of protein-peptide binding residues are essential for protein-peptide interactions and advancing drug discovery. To address this problem, extensive research efforts have been made to design more discriminative feature representations. However, extracting these explicit features usually depend on third-party tools, resulting in low computational efficacy and suffering from low predictive performance. In this study, we design an end-to-end deep learning-based method, E2EPep, for protein-peptide binding residue prediction using protein sequence only. E2EPep first employs and fine-tunes two state-of-the-art pre-trained protein language models that can extract two different high-latent feature representations from protein sequences relevant for protein structures and functions. A novel feature fusion module is then designed in E2EPep to fuse and optimize the above two feature representations of binding residues. In addition, we have also design E2EPep+, which integrates E2EPep and PepBCL models to improve the prediction performance. Experimental results on two independent testing data sets demonstrate that E2EPep and E2EPep+ could achieve the average AUC values of 0.846 and 0.842 while achieving an average Matthew’s correlation coefficient value that is significantly higher than that of existing most of sequence-based methods and comparable to that of the state-of-the-art structure-based predictors. Detailed data analysis shows that the primary strength of E2EPep lies in the effectiveness of feature representation using cross-attention mechanism to fuse the embeddings generated by two fine-tuned protein language models. E2EPep independent operation of the package can be obtained at https://github.com/ckx259/E2EPep.git for academic use only.

## Pre-requisite:
    - Python3.7, Anaconda3
    - Linux system

## Environmental preparation
* If you have  met the conditions in Pre-requisite, ignore the following operations. If your version of Python is not 3.7, do the following.

~~~
conda create -n E2EPep python==3.7
conda activate E2EPep
~~~

## Installation:

* Download this repository at https://github.com/jun-csbio/e2epep/. Then, uncompress it and run the following command lines on Linux System.

~~~
  $ cd E2EPep-master
  $ chmod 777 ./install.sh
  $ ./install.sh
~~~

* If the package cannot work correctly on your computational cluster, you should install the dependencies via running the following commands:

~~~
  $ cd E2EPep-main
  $ pip install -r requirements.txt
~~~

## What else do you need to do
* you need to download pytorch_model.bin file from the following URL https://huggingface.co/Rostlab/prot_bert_bfd/blob/main/pytorch_model.bin. And put pytorch_model.bin file into premodel/prot_bert_bfd directory.


* PepBCL model can be downloaded at https://github.com/Ruheng-W/PepBCL. And put the downloaded model, Dataset1_AUC_0.815080211458067,MCC_0.38696326734790976.pl, into the model/PepBCL_model directory

## Run example
* The prediction command of E2EPep
~~~
  $ python predict.py -sf example/results_of_voting/ -seq_fa example/seq.fa
~~~

* The prediction command of E2EPep+ by using single fold model ("-n" represent using nth fold model, n can be 0,1,2,3, or 4)

~~~
  $ python predictPlus.py -sf example/resultsPlus_of_single_fold_model/ -seq_fa example/seq.fa -n 0
~~~


## Result
* The prediction result file (e.g., "1dpuA.pred") of each protein (e.g., 1dpuA) in your input fasta file (-seq_fa) could be found in the folder which you input as "-sf".

* There are four columns in each prediction result file. The 1st column is the residue index. The 2nd column is the residue type. The 3th column is the probability predicted to be peptide-binding residues. The 4th column is the prediction result ('B' and 'N' mean the predicted Peptide-binding and non-Peptide-binding residue, respectively).   For example:
~~~
Index   AA      Prob[threshold]     State
0       A       0.199   N
1       N       0.278   N
2       G       0.146   N
3       L       0.291   N
4       T       0.146   N
5       V       0.370   N
6       A       0.100   N
7       Q       0.332   N
8       N       0.324   N
9       Q       0.221   N
10      V       0.086   N
11      L       0.212   N
12      N       0.275   N
13      L       0.095   N
14      I       0.092   N
15      K       0.135   N
16      A       0.101   N
17      C       0.091   N
18      P       0.137   N
19      R       0.155   N
20      P       0.231   N
21      E       0.224   N
22      G       0.549   N
23      L       0.123   N
24      N       0.212   N
25      F       0.417   N
26      Q       0.251   N
27      D       0.132   N
28      L       0.074   N
29      K       0.426   N
30      N       0.178   N
31      Q       0.166   N
32      L       0.137   N
33      K       0.326   N
34      H       0.252   N
35      M       0.164   N
36      S       0.243   N
37      V       0.314   N
38      S       0.293   N
39      S       0.134   N
40      I       0.127   N
41      K       0.759   B
42      Q       0.534   N
43      A       0.089   N
44      V       0.312   N
45      D       0.817   B
46      F       0.761   B
47      L       0.607   N
48      S       0.871   B
49      N       0.855   B
50      E       0.767   B
51      G       0.832   B
52      H       0.815   B
53      I       0.257   N
54      Y       0.913   B
55      S       0.883   B
56      T       0.918   B
57      V       0.858   B
58      D       0.805   B
59      D       0.816   B
60      D       0.624   N
61      H       0.264   N
62      F       0.354   N
63      K       0.733   B
64      S       0.093   N
65      T       0.786   B
66      D       0.407   N
67      A       0.261   N
68      E       0.191   N
~~~

## Tips
* <b>This package is only free for academic use</b>. If you have any question, please email Jun Hu: junh_cs@126.com

## References
[1] . Protein-peptide binding residue prediction based on protein language models and feature fusion method.

