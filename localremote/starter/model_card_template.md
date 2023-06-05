# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a prediction model for the salary range of persons, based on the census data Adult Data Set
available at
http://archive.ics.uci.edu/ml/datasets/adult
A simple model was chosen to learn about API develpment of that model

LogisticRegression. Libraries used:
- numpy==1.24.3
- pandas==2.0.2
- scikit-learn==1.2.2
## Intended Use
To predict the income category >50K, <=50K based on these variables:
- age: continuous.
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- fnlwgt: continuous.
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- education-num: continuous.
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Female, Male.
- capital-gain: continuous.
- capital-loss: continuous.
- hours-per-week: continuous.
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.


## Training Data
It was used with the hole dataset, with train_test_split(data, test_size=0.20), with preprocessing 
- OneHotEncoder(sparse=False, handle_unknown="ignore")
- LabelBinarizer()


## Evaluation Data
Used the test data obtained from train_test_split explained previously

## Metrics
3 metrics are chosen, and in the model saved and used in the project these are the metrics
obtained
precision: 0.7317073170731707
recall: 0.27932960893854747
fbeta: 0.4043126684636118

## Ethical Considerations
A simple bias analysis was done by analyzing the change of the 3 metrics described previously across the different values of the columns of the dataset. The
resulting data was saved at ./localremote/starter/slice_output.txt. As an example, these are
across the race column, where we can see a difference in precision of the model for White and Black categories.

--------------------------------------
race White
precision,0.7416974169741697
--------------------------------------
race Black
precision,0.6052631578947368


## Caveats and Recommendations

- The dataset is from 1994, so this is inly for educaitonal purposes. A similar approach with newer data and a new threshold (because the 50K limit should
change due to inflation)
- A simple model was used to ilustrative processes, a more complex model such as a randomforest should consider overfitting considerations, as it might
be better with bias, however it could be overfitting on the data
