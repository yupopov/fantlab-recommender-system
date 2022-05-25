# Fantlab recommender system
This is a project aimed at creating recommendation system for russian e-library fantlab.ru. At the moment there are some datasets, python modules to create them and [LightFM](making.lyst.com/lightfm/docs/home.html) collaborative filtration model, showing decent results. Also a content-based model will be available soon.

# Repository content

Here is a little repository content guide.

### [Data/raw](github.com/yupopov/fantlab-recommender-system/tree/main/data/raw)

Data/raw folder contains: 
- parsed work ids
- works information file
- work features file

### [Data/interim](github.com/yupopov/fantlab-recommender-system/tree/main/data/interim )

This is a folder which contains all the information used during the model fitting, inference etc.:
- embeddings obtained by different ways stored in torch tensor/np.array/sparse matrix
- work descriptions, raw and prepared to consume by all the models, assembled to a dictionary
- key to index dictionary in case of using work descriptions as a list, not dictionary

### [src/data_retrieval](github.com/yupopov/fantlab-recommender-system/tree/main/src/data_retrieval)

Folder which is needed to obtain data from Fantlab public API, contains:
- html parser to extract work ids (yeah, they were obtained from html's of "the most" from every form section, there obviously is a smarter way to do this)
- asynchronous downloaders of work infos and users marks

### [src/models](github.com/yupopov/fantlab-recommender-system/tree/main/src/models)

Folder containing models modules:
- [LinearRecommender.py](github.com/yupopov/fantlab-recommender-system/blob/main/src/models/LinearRecommender.py) module which eats user interaction matrix and work embeddings and builds personal content recommendation for every user 
- [lightfm.py](github.com/yupopov/fantlab-recommender-system/blob/main/src/models/lightfm.py) custom recommend function 
- [get_top_k_predictions_with_label.py](github.com/yupopov/fantlab-recommender-system/blob/main/src/models/get_top_k_predictions_with_label.py) auxiliary module to get predictions from the model 
### [src/preprocessing](github.com/yupopov/fantlab-recommender-system/tree/main/src/preprocessing)

Folder containing data preprocessing modules:
- [datasets.py](github.com/yupopov/fantlab-recommender-system/blob/main/src/preprocessing/datasets.py) has functions to create datasets for both collaborative and content models
- [item_features_buildup.py](github.com/yupopov/fantlab-recommender-system/blob/main/src/preprocessing/item_features_buildup.py) module is needed to build additional categorical features of works put in consumable by LightFM format
- [mark_weights.py](github.com/yupopov/fantlab-recommender-system/blob/main/src/preprocessing/mark_weights.py) module is needed to filter interaction matrix by marks
- [time_weights.py](github.com/yupopov/fantlab-recommender-system/blob/main/src/preprocessing/time_weights.py) module is needed to filter interaction matrix by time
- [title_parser.py](github.com/yupopov/fantlab-recommender-system/blob/main/src/preprocessing/title_parser.py) contains methods to parse work infos in .json format to pd.DataFrame