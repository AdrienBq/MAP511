params={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 128,
    'enc_nh': 512,
    'dec_nh': 512,
    'log_niter': 50,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'batch_size': 32,
    'epochs': 50,
    'test_nepoch': 5,
    'train_data': 'datasets/tatoeba_data/tatoeba.train.txt',
    'val_data': 'datasets/tatoeba_data/tatoeba.valid.txt',
    'test_data': 'datasets/tatoeba_data/tatoeba.test.txt',
    'ais_prior': 'normal',
    'ais_T': 500,
    'ais_K': 3,
    "label": False
}
