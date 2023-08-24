class Config(object):	
	apr_dir = 'model/'
	data_dir = 'datasets_NER/'
	model_name = 'best_model.pt'
	epoch = 15
	bert_model = 'bert-base-uncased'
	lr = 3e-5
	eps = 1e-10
	batch_size = 16
	mode = 'prediction' # for prediction mode = "prediction"
	training_data = 'train.tsv'
	val_data = 'devel.tsv'
	test_data = 'test.tsv'
	test_out = 'test_prediction.csv'
	raw_prediction_output = 'raw_prediction.csv'
