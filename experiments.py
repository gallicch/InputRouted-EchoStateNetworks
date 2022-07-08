from RC import *

import tensorflow as tf
from tensorflow import keras
from keras_tuner import BayesianOptimization

import os
import pickle
from time import time, localtime, strftime

#GPU setup: no GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


#common experimental setting:
num_epochs = 500 #maximum number of epochs
num_guesses = 5 #repetitions of each experiment (for the selected configurations)
patience = 10 #patience for the early stopping callback
max_units = 200 #this is the number of units used in the experiments
batch_size = 32
max_trials = 300 #number of explored configurations by the Bayesian Search algorithm

dataset_names = [
                 'CBF3','CBF2','CBF1','CBF0','CBF', #the synthetic datasets (CBFn contains n+1 noisy input dimensions)
                 'BasicMotions', #the others are the real-world datasets ...
                 'CharacterTrajectories',
                 'Epilepsy',
                 'Libras',   
                 'PhonemeSpectra',
                 'UWaveGestureLibrary'
                ]

for dataset_name in dataset_names:


    #load the dataset and setup the paths
    root_path = './'
    datasets_path = os.path.join(root_path,'datasets')
    keras_datasets_path = os.path.join(datasets_path)
    model_selection_path = os.path.join(root_path,'model_selection_data')

    keras_dataset_filename = os.path.join(keras_datasets_path,dataset_name+'_dataset.p')
    dataset = pickle.load(open(keras_dataset_filename,"rb"))
    x_train_all,y_train_all,x_test, y_test,x_train, x_val, y_train, y_val = dataset[0],dataset[1],dataset[2],dataset[3],dataset[4],dataset[5],dataset[6],dataset[7]


    #dataset-specific experimental setting:
    output_units = max(y_train_all)+1
    if output_units == 2:
        output_units = 1
        output_function = 'sigmoid'
        loss_function = 'binary_crossentropy'
    else:
        output_activation = 'softmax'
        loss_function = 'sparse_categorical_crossentropy'

    #sets the number of sub-reservoirs as the number of input dimensions
    num_sub_reservoirs = x_train.shape[-1]

    results_path = os.path.join(root_path, 'results_experiments',dataset_name)
    #create the results path if it does not exists
    if not os.path.exists(results_path):
        os.makedirs(results_path)



    #experiments with the IRESN model
    model_type = 'IRESN'

    model_selection_times_filename = os.path.join(results_path,'model_selection_times_'+model_type+'.p')
    times_filename = os.path.join(results_path,'times_'+model_type+'.p')
    accuracy_filename = os.path.join(results_path,'accuracy_'+model_type+'.p')
    tuner_filename = os.path.join(results_path,'tuner_'+model_type+'.p')

    results_logger_filename = os.path.join(results_path,'results_logger_'+model_type+'.txt')

    tf.random.set_seed(42)
    def build_model_IRESN(hp):
        model = IRESN(units = max_units,
                    output_units = output_units, 
                    output_activation = output_activation,
                    leaky = hp.Choice('leaking_rate', values = [0.01, 0.1, 1.], default = 0.01),                  
                    input_scaling = [hp.Choice('input_scaling_'+str(i), values = [0., 0.001, 0.01, 0.1, 1.], default = 1.) for i in range(num_sub_reservoirs)],
                    inter_scaling = [hp.Choice('inter_scaling'+str(i), values = [0., 0.001, 0.01, 0.1, 1.], default = 1.) for i in range(num_sub_reservoirs)],
                    bias_scaling = [hp.Choice('bias_scaling'+str(i), values = [0., 0.001, 0.01, 0.1, 1.], default = 1.) for i in range(num_sub_reservoirs)],
                    spectral_radius = [hp.Choice('spectral_radius'+str(i), values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1], default = 0.9) for i in range(num_sub_reservoirs)])
        model.readout.compile(
            optimizer=keras.optimizers.Adam(),
            loss=loss_function,
            metrics=['accuracy'])

        return model


    tuner = BayesianOptimization(
        build_model_IRESN,
        objective='val_accuracy',
        max_trials = max_trials,
        directory=os.path.join(model_selection_path,dataset_name,'IRESN_ms'),
        project_name='ESANN22',
        overwrite = True,
        seed = 42)

    results_logger = open(results_logger_filename,'w')
    results_logger.write('Experiment with '+model_type+' on dataset '+ dataset_name + ' starting now\n')
    time_string_start = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('** local time = '+ time_string_start+'\n')

    initial_model_selection_time = time()
    tuner.search(x_train, y_train,
                 epochs=num_epochs,
                 validation_data = (x_val,y_val),
                 batch_size = batch_size,
                 callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience)]
                )
    elapsed_model_selection_time = time()-initial_model_selection_time

    time_string = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('model selection concluded at local time = '+ time_string+'\n')


    #choose the best hyper-parameters
    best_model_hp = tuner.get_best_hyperparameters()[0]
    acc_ts = []
    required_time = []
    tf.random.set_seed(42)
    for i in range(num_guesses):
      initial_time = time()
      model = tuner.hypermodel.build(best_model_hp)
      model.fit(x_train, y_train, epochs = num_epochs, batch_size = batch_size, validation_data = (x_val, y_val),callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience, restore_best_weights = True)])
      _, acc = model.evaluate(x_test,y_test)
      required_time.append(time()-initial_time)
      acc_ts.append(acc)

    time_string_end = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('*** best model assessment concluded at local time = '+ time_string_end+'\n')


    with open(model_selection_times_filename, 'wb') as f:
        pickle.dump(elapsed_model_selection_time, f)
    with open(times_filename, 'wb') as f:
        pickle.dump(required_time, f)
    with open(accuracy_filename, 'wb') as f:
        pickle.dump(acc_ts, f)    
    with open(tuner_filename, 'wb') as f:
        pickle.dump(tuner, f)     

    print('--'+model_type+' on {}--'.format(dataset_name))
    print('Results: MEAN {} STD {}'.format(np.mean(acc_ts),np.std(acc_ts)))
    print('----- required time: MEAN {} STD {}'.format(np.mean(required_time),np.std(required_time)))
    print('----- total model selection time: {}'.format(elapsed_model_selection_time))



    results_logger.write('** Results:\n')
    results_logger.write('Start time: '+time_string_start+'\n')
    results_logger.write('End time: '+time_string_end+'\n')
    results_logger.write('Accuracy: MEAN {} STD {}\n'.format(np.mean(acc_ts),np.std(acc_ts)))
    results_logger.write('Model selection time: {} seconds = {} minutes\n'.format(elapsed_model_selection_time, elapsed_model_selection_time/60))
    results_logger.write('Average time for TR,TS: MEAN {} STD {}\n'.format(np.mean(required_time),np.std(required_time)))
    results_logger.write('Model summary:\n')


    model.readout.summary(print_fn=lambda x: results_logger.write(x + '\n'))
    tuner.results_summary(1)
    results_logger.close()


    #experiments with the IRESN (base) model
    model_type = 'IRESN_base'

    model_selection_times_filename = os.path.join(results_path,'model_selection_times_'+model_type+'.p')
    times_filename = os.path.join(results_path,'times_'+model_type+'.p')
    accuracy_filename = os.path.join(results_path,'accuracy_'+model_type+'.p')
    tuner_filename = os.path.join(results_path,'tuner_'+model_type+'.p')

    results_logger_filename = os.path.join(results_path,'results_logger_'+model_type+'.txt')

    tf.random.set_seed(42)
    def build_model_IRESN_base(hp):
        #the same code as for the full IRESN is used here
        #just notice that each hyper-parameter is set to the same value for all the sub-reservoirs
        model = IRESN(units = max_units,
                    output_units = output_units, 
                    output_activation = output_activation,
                    leaky = hp.Choice('leaking_rate', values = [0.01, 0.1, 1.], default = 0.01),                  
                    input_scaling = hp.Choice('input_scaling', values = [0., 0.001, 0.01, 0.1, 1.], default = 1.) *np.ones(shape = (num_sub_reservoirs,)),
                    inter_scaling = hp.Choice('inter_scaling', values = [0., 0.001, 0.01, 0.1, 1.], default = 1.) *np.ones(shape = (num_sub_reservoirs,)),
                    bias_scaling = hp.Choice('bias_scaling', values = [0., 0.001, 0.01, 0.1, 1.], default = 1.) *np.ones(shape = (num_sub_reservoirs,)),
                    spectral_radius = hp.Choice('spectral_radius', values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1], default = 0.9) *np.ones(shape = (num_sub_reservoirs,)))
        model.readout.compile(
            optimizer=keras.optimizers.Adam(),
            loss=loss_function,
            metrics=['accuracy'])

        return model


    tuner = BayesianOptimization(
        build_model_IRESN_base,
        objective='val_accuracy',
        max_trials = max_trials,
        directory=os.path.join(model_selection_path,dataset_name,'IRESN_ms_base'),
        project_name='ESANN22',
        overwrite = True,
        seed = 42)

    results_logger = open(results_logger_filename,'w')
    results_logger.write('Experiment with '+model_type+' on dataset '+ dataset_name + ' starting now\n')
    time_string_start = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('** local time = '+ time_string_start+'\n')

    initial_model_selection_time = time()
    tuner.search(x_train, y_train,
                 epochs=num_epochs,
                 validation_data = (x_val,y_val),
                 batch_size = batch_size,
                 callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience)]
                )
    elapsed_model_selection_time = time()-initial_model_selection_time

    time_string = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('model selection concluded at local time = '+ time_string+'\n')


    #choose the best hyper-parameters
    best_model_hp = tuner.get_best_hyperparameters()[0]
    acc_ts = []
    required_time = []
    tf.random.set_seed(42)
    for i in range(num_guesses):
      initial_time = time()
      model = tuner.hypermodel.build(best_model_hp)
      model.fit(x_train, y_train, epochs = num_epochs, batch_size = batch_size, validation_data = (x_val, y_val),callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience, restore_best_weights = True)])
      _, acc = model.evaluate(x_test,y_test)
      required_time.append(time()-initial_time)
      acc_ts.append(acc)

    time_string_end = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('*** best model assessment concluded at local time = '+ time_string_end+'\n')


    with open(model_selection_times_filename, 'wb') as f:
        pickle.dump(elapsed_model_selection_time, f)
    with open(times_filename, 'wb') as f:
        pickle.dump(required_time, f)
    with open(accuracy_filename, 'wb') as f:
        pickle.dump(acc_ts, f)    
    with open(tuner_filename, 'wb') as f:
        pickle.dump(tuner, f)     

    print('--'+model_type+' on {}--'.format(dataset_name))
    print('Results: MEAN {} STD {}'.format(np.mean(acc_ts),np.std(acc_ts)))
    print('----- required time: MEAN {} STD {}'.format(np.mean(required_time),np.std(required_time)))
    print('----- total model selection time: {}'.format(elapsed_model_selection_time))



    results_logger.write('** Results:\n')
    results_logger.write('Start time: '+time_string_start+'\n')
    results_logger.write('End time: '+time_string_end+'\n')
    results_logger.write('Accuracy: MEAN {} STD {}\n'.format(np.mean(acc_ts),np.std(acc_ts)))
    results_logger.write('Model selection time: {} seconds = {} minutes\n'.format(elapsed_model_selection_time, elapsed_model_selection_time/60))
    results_logger.write('Average time for TR,TS: MEAN {} STD {}\n'.format(np.mean(required_time),np.std(required_time)))
    results_logger.write('Model summary:\n')


    model.readout.summary(print_fn=lambda x: results_logger.write(x + '\n'))
    tuner.results_summary(1)
    results_logger.close()    
    
    #experiments with the ESN model
    model_type = 'ESN'

    model_selection_times_filename = os.path.join(results_path,'model_selection_times_'+model_type+'.p')
    times_filename = os.path.join(results_path,'times_'+model_type+'.p')
    accuracy_filename = os.path.join(results_path,'accuracy_'+model_type+'.p')
    tuner_filename = os.path.join(results_path,'tuner_'+model_type+'.p')

    results_logger_filename = os.path.join(results_path,'results_logger_'+model_type+'.txt')

    tf.random.set_seed(42)
    def build_model_ESN(hp):
        model = ESN(units = max_units,
                    output_units = output_units, 
                    output_activation = output_activation,
                    input_scaling = hp.Choice('input_scaling', values = [0., 0.001, 0.01, 0.1, 1.], default = 1.),
                    bias_scaling = hp.Choice('bias_scaling', values = [0., 0.001, 0.01, 0.1, 1.], default = 1.),
                    leaky = hp.Choice('leaking_rate', values = [0.01, 0.1, 1.], default = 0.01),
                    spectral_radius = hp.Choice('spectral_radius',  values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1], default = 0.9))
        model.readout.compile(
            optimizer=keras.optimizers.Adam(),
            loss=loss_function,
            metrics=['accuracy'])

        return model


    tuner = BayesianOptimization(
        build_model_ESN,
        objective='val_accuracy',
        max_trials = max_trials,
        directory=os.path.join(model_selection_path,dataset_name,'ESN_ms'),
        project_name='ESANN22',
        overwrite = True,
        seed = 42)

    results_logger = open(results_logger_filename,'w')
    results_logger.write('Experiment with '+model_type+' on dataset '+ dataset_name + ' starting now\n')
    time_string_start = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('** local time = '+ time_string_start+'\n')

    initial_model_selection_time = time()
    tuner.search(x_train, y_train,
                 epochs=num_epochs,
                 validation_data = (x_val,y_val),
                 batch_size = batch_size,
                 callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience)]
                )
    elapsed_model_selection_time = time()-initial_model_selection_time

    time_string = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('model selection concluded at local time = '+ time_string+'\n')


    #choose the best hyper-parameters
    best_model_hp = tuner.get_best_hyperparameters()[0]
    acc_ts = []
    required_time = []
    tf.random.set_seed(42)
    for i in range(num_guesses):
      initial_time = time()
      model = tuner.hypermodel.build(best_model_hp)
      model.fit(x_train, y_train, epochs = num_epochs, batch_size = batch_size, validation_data = (x_val, y_val),callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience = patience, restore_best_weights = True)])
      _, acc = model.evaluate(x_test,y_test)
      required_time.append(time()-initial_time)
      acc_ts.append(acc)

    time_string_end = strftime("%Y/%m/%d %H:%M:%S", localtime())
    results_logger.write('*** best model assessment concluded at local time = '+ time_string_end+'\n')


    with open(model_selection_times_filename, 'wb') as f:
        pickle.dump(elapsed_model_selection_time, f)
    with open(times_filename, 'wb') as f:
        pickle.dump(required_time, f)
    with open(accuracy_filename, 'wb') as f:
        pickle.dump(acc_ts, f)    
    with open(tuner_filename, 'wb') as f:
        pickle.dump(tuner, f)     

    print('--'+model_type+' on {}--'.format(dataset_name))
    print('Results: MEAN {} STD {}'.format(np.mean(acc_ts),np.std(acc_ts)))
    print('----- required time: MEAN {} STD {}'.format(np.mean(required_time),np.std(required_time)))
    print('----- total model selection time: {}'.format(elapsed_model_selection_time))



    results_logger.write('** Results:\n')
    results_logger.write('Start time: '+time_string_start+'\n')
    results_logger.write('End time: '+time_string_end+'\n')
    results_logger.write('Accuracy: MEAN {} STD {}\n'.format(np.mean(acc_ts),np.std(acc_ts)))
    results_logger.write('Model selection time: {} seconds = {} minutes\n'.format(elapsed_model_selection_time, elapsed_model_selection_time/60))
    results_logger.write('Average time for TR,TS: MEAN {} STD {}\n'.format(np.mean(required_time),np.std(required_time)))
    results_logger.write('Model summary:\n')


    model.readout.summary(print_fn=lambda x: results_logger.write(x + '\n'))
    tuner.results_summary(1)
    results_logger.close()
"""