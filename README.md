# Using Neural Networks for time series prediction
In this project, I have applied CNN, RNN, TensorFlow, Keras using Python, implemented basic neural models for the data, and then fine tuned on the selected models.

## Neural Network Architecture Experiment
After Data Importation & Preprocessing like scaling and feature selection, 24-lag Supervised Data Transformation & Train-Test Split, I wanted to try a set of basic models on original dataset to see which one performs better and worth further investigation. I could use a subset of features in the beginning as for my final model, but it would risk information loss in the beginning and could make it hard to find out which model performs better.
![model exploration](https://github.com/cathyhuangli/Deep-Learning-Exploration-on-Weather-Data-Project/blob/master/Neural%20Network%20Models%20exploration.png)

## Hyperparameter fine-tuning
Based on the base models results from Table1, I chose model_4 as the model to investigate further and perform parameter experiments in Table 2. I have also tried the second best model, which is the base model to perform further investigation, but the performance is very similar as model_4, not better, so I focused on model_4 for the experiments.
![model tweaking phase 1](https://github.com/cathyhuangli/Deep-Learning-Exploration-on-Weather-Data-Project/blob/master/model%20tweaking%20phase%201.png)

## Final Model fine_tuning and feature selection Experiment
Based on the best model selected so far, model_4, I tried to firstly tune epochs and batch_sizes, I have also tried batch-sized in 80,300,400, but none of them perform as good as 200. And when using the original data, increasing of epoch size from 50 to 100 did not increase the performance; while using selected features, the result is improved. The selected features are based on the plots and correlation table of the 14 features, and there are four features that seems not highly correlated with our target feature, 'p (mbar)','wv (m/s)', 'max. wv (m/s)','wd' with correlation less than 0.1. That is why I tried to fit my best model using a reduced dataset with 10 selected variables, and the result is the best I have achieved so far.
![model tweaking phase 2](https://github.com/cathyhuangli/Deep-Learning-Exploration-on-Weather-Data-Project/blob/master/model%20tweaking%20phase%202.png)


## plot of MAE loss on the best two models
From the plots, the MAE loss of best model using all feature and selected features below
show that the selected 10 features have created a more smoothed dataset for time series
prediction and thus achieved better result than the model using all features. As we can see from
the plots, the loss plots for training datasets are very similar; while for test dataset, there is huge
fluctuation in the plot using all features, and the loss plot using selected features are much more
stable over the 100 epochs performance. Overall, both models showed a downward trend of loss
toward 0 as the epoch increases.
![plot of MAE loss](https://github.com/cathyhuangli/Deep-Learning-Exploration-on-Weather-Data-Project/blob/master/Plot%20of%20best%20two%20models.png)


