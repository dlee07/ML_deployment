from feature_engine.encoding import OneHotEncoder, RareLabelEncoder

# for imputation
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer

titanic_pipe = Pipeline(
	[
		(
            "categorical_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.model_config.categorical_vars,
            ),
        ),

        ('median_imputation', MeanMedianImputer(
        imputation_method='median', variables=config.model_config.numerical_vars)),

        # Extract letter from cabin
        ('extract_letter', ExtractLetterTransformer(variables=config.model_config.cabin_vars)),


        # encode categorical variables using one hot encoding into k-1 variables
        ('categorical_encoder', OneHotEncoder(
        drop_last=True, variables=config.model_config.categorical_vars)),

        # scale
        ('scaler', StandardScaler()),

        ('RF', RandomForestClassifier(random_state=0))
	]
)