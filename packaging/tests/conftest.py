import logging

import pytest
from sklearn.model_selection import train_test_split

from classification_model.config.core import config
from classification_model.processing.data_manager import load_dataset

@pytest.fixture

def sample_input_data():

	data = load_dataset(file_name=config.app_config.raw_data_file)

	X_train, X_test, y_train, y_test = train_test_split(
	    data,  # predictors
	    data[config.model_config.target],
	    test_size=config.model_config.test_size,
	    # we are setting the random seed here for reproducibility
	    random_state=config.model_config.random_state
	)
	return X_test