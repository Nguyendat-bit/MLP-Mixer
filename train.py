from data import DataLoader
from model import *
from argparse import ArgumentParser
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow as tf
from tensorflow.keras.callbacks import *
import sys
tf.config.experimental_run_functions_eagerly(True)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--batch-size', default= 16, type= int)
    parser.add_argument('--train-folder', type= str, required= True)
    parser.add_argument('--valid-folder', type= str, default= None)
    parser.add_argument('--epochs', default= 100, type= int)
    parser.add_argument('--classes', default= 5, type= int)
    parser.add_argument('--lr', default= 0.07, type= float)
    parser.add_argument('--shuffle', default= True, type= bool)
    parser.add_argument('--augmented', default= False, type= bool)
    parser.add_argument('--seed', default= 2021, type= int)
    parser.add_argument('--image-size', default= 150, type= int)
    parser.add_argument('--mixer-layer', default= 8, type= int )
    parser.add_argument('--patch-size', default= 32, type= int)
    parser.add_argument('--hidden-size', default= 512, type= int)
    parser.add_argument('--Dc', default= 2048, type= int)
    parser.add_argument('--Ds', default= 256, type = int )
    parser.add_argument('--dropout', default= 0.0, type= float)
    parser.add_argument('--optimizer', default= 'adam', type= str)
    parser.add_argument('--model-save', default= 'mlp_mixer.h5', type= str)
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    print('---------------------Welcome to MLP-Mixer-------------------')
    print('Author')
    print('Github: Nguyendat-bit')
    print('Email: nduc0231@gmail')
    print('---------------------------------------------------------------------')
    print('Training MLP-Mixer model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    

    # Load Data
    print("-------------LOADING DATA------------")
    datasets = DataLoader(args.train_folder, args.valid_folder, augment= args.augmented, seed= args.seed, batch_size= args.batch_size, shuffle= args.shuffle, image_size= (args.image_size, args.image_size))
    train_data, val_data = datasets.build_dataset()
    num_classes  = len(train_data.class_indices)
    print(f"num-classes: {num_classes}")
    # Initializing models
    mlp_mixer = MLP_Mixer(
        inp_size= (args.image_size, args.image_size, 3), 
        classes= num_classes, 
        mixer_layer= args.mixer_layer,
        patch_size= args.patch_size,
        C = args.hidden_size,
        Dc= args.Dc,
        Ds= args.Ds,
        dropout= args.dropout).build()
    mlp_mixer.summary()
    # Set up loss function
    loss = CategoricalCrossentropy()

    # Optimizer Definition
    if args.optimizer == 'adam':
        optimizer = Adam(learning_rate=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = SGD(learning_rate=args.lr)
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=args.lr)
    elif args.optimizer == 'adadelta':
        optimizer = Adadelta(learning_rate=args.lr)
    elif args.optimizer == 'adamax':
        optimizer = Adamax(learning_rate=args.lr)
    elif args.optimizer == 'adagrad':
        optimizer = Adagrad(learning_rate= args.lr)
        
    else:
        raise 'Invalid optimizer. Valid option: adam, sgd, rmsprop, adadelta, adamax, adagrad'

    # Callback
    if val_data == None:
        checkpoint = ModelCheckpoint(args.model_save, monitor= 'acc', save_best_only=  True, verbose = 1)
    else:
        checkpoint = ModelCheckpoint(args.model_save, monitor= 'val_acc', save_best_only=  True, verbose = 1) 
    lr_R = ReduceLROnPlateau(monitor= 'acc', patience= 3, verbose= 1 , factor= 0.5, min_lr= 0.00001)

    # Complie optimizer and loss function into model
    mlp_mixer.compile(optimizer= optimizer, loss= loss, metrics= ['acc'])

    # Training model 
    print('-------------Training Mobilenet_V2------------')
    mlp_mixer.fit(train_data, validation_data= val_data, epochs= args.epochs, verbose= 1, callbacks= [checkpoint, lr_R])


