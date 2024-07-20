# Toma-Machine Learning is All You Need: A Simple Token-based Approach for Effective Code Clone Detection

## Project Structure  
  
```shell  
Toma  
|-- Tokenization_and_FeatureExtraction.py     	// implement the first two phases:  Tokenization phase and Feature Extraction phase
|-- Classification.py   // implement the Classification phase  
```

### Tokenization_and_FeatureExtraction.py
- Input: dataset with source codes
- Output: feature vectors of code pairs 
```
python Tokenization_and_FeatureExtraction.py
```

### classification.py
- Input: feature vectors of dataset
- Output: recall, precision, and F1 scores of machine learning algorithm
```
python classification.py
```

## Dataset

- The zip id2sourcecode of the dataset folder (https://github.com/CGCL-codes/Toma/tree/main/dataset) can be unzipped to get the Java function files in the BCB dataset. 

- type1-type5 CSV files correspond to Type-1, Type-2, Strongly Type-3, Moderately Type-3, Weakly Type-3/Type-4, which are described in the dataset section of the paper.

- clone.csv is a collection of clone pairs for the 5 clone types and nonclone.csv is a collection of all non-clone pairs. 

- For all CSV files you only need to focus on the first two columns, i.e. the two function ids of the code pairs (clone pairs or non-clone pairs), according to which you can find the corresponding Java functions in the id2sourcecode folder.


## Collinearity of Features

![Image text](https://github.com/TomaCodes/Toma/blob/main/collinearity.jpg)


# Publication
Siyue Feng, Wenqi Suo, Yueming Wu, Deqing Zou, Yang Liu, and Hai Jin. 2024. Machine Learning is All You Need: A Simple Token-based Approach for Effective Code Clone Detection. In 2024 IEEE/ACM 46th International Conference on Software Engineering (ICSE ’24), April 14–20, 2024, Lisbon, Portugal. ACM, New York, NY, USA, 13 pages. https://doi.org/10.1145/3597503.3639114

If you use our dataset or source code, please kindly cite our paper:
```
@INPROCEEDINGS{toma2024,
  author={Feng, Siyue and Suo, Wenqi and Wu, Yueming and Zou, Deqing and Liu, Yang and Jin, Hai},
  booktitle={2024 IEEE/ACM 46th International Conference on Software Engineering (ICSE ’24)}, 
  title={Machine Learning is All You Need: A Simple Token-based Approach for Effective Code Clone Detection}, 
  year={2024},
  doi={10.1145/3597503.3639114}}
```

# Support or Contact
Toma is developed in the National Engineering Research Center for Big Data Technology and System, Services Computing Technology and System Lab, Hubei Key Laboratory of Distributed System Security, Hubei Engineering Research Center on Big Data Security, Cluster and Grid Computing Lab, Huazhong University of Science and Technology, Wuhan, China by Siyue Feng (fengsiyue@hust.edu.cn), Wenqi Suo (suowenqi@hust.edu.cn), Yueming Wu (wuyueming21@gmail.com), Deqing Zou (deqingzou@hust.edu.cn), Yang Liu (yangliu@ntu.edu.sg), and Hai Jin (hjin@hust.edu.cn).
