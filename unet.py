from keras import backend as K
from common_defs import *
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import *
# a dict with x_train, y_train, x_test, y_test
from load_data import data
from hyperopt import hp
from hyperopt.pyll.stochastic import sample

im_height = 128
im_width = 128

space = {
	'batch_size': hp.choice('bs',(16, 32, 64, 128, 256)),
	'optimizer': hp.choice( 'o', ( 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'sgd'))		
}
def print_params( params ):
	print("batch_size:", params['batch_size'])
	print("\noptimizer:", params['optimizer'])	


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def get_params():

	params = sample( space )
	return handle_integers( params )

def try_params( n_iterations, params ):
	
	print ("iterations:", n_iterations)
	print_params( params )
	
	x_train = data['x_train']
	x_test = data['x_test']
	y_train = data['y_train']
	y_test = data['y_test']

	inputs = Input((None, None, 1))
	c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
	c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
	p1 = MaxPooling2D((2, 2)) (c1)

	c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
	c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
	p2 = MaxPooling2D((2, 2)) (c2)

	c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
	c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
	p3 = MaxPooling2D((2, 2)) (c3)

	c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
	c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
	p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

	c5 = Conv2D(64, (3, 3), activation='relu', padding='same') (p4)
	c5 = Conv2D(64, (3, 3), activation='relu', padding='same') (c5)
	p5 = MaxPooling2D(pool_size=(2, 2)) (c5)

	c55 = Conv2D(128, (3, 3), activation='relu', padding='same') (p5)
	c55 = Conv2D(128, (3, 3), activation='relu', padding='same') (c55)

	u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c55)
	u6 = concatenate([u6, c5])
	c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
	c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

	u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
	u71 = concatenate([u71, c4])
	c71 = Conv2D(32, (3, 3), activation='relu', padding='same') (u71)
	c61 = Conv2D(32, (3, 3), activation='relu', padding='same') (c71)

	u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c61)
	u7 = concatenate([u7, c3])
	c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
	c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

	u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
	u8 = concatenate([u8, c2])
	c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
	c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

	u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
	u9 = concatenate([u9, c1], axis=3)
	c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
	c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

	outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

	model = Model(inputs=[inputs], outputs=[outputs])
	model.compile(optimizer=params['optimizer'], loss='binary_crossentropy', metrics=[dice_coef])
	model.summary()
	validation_data = ( x_test, y_test )

	early_stopping = EarlyStopping( monitor = 'val_loss', patience = 5, verbose = 0 )
	history = model.fit(
		x_train, 
		y_train,  
		batch_size = params['batch_size'], 
		epochs = int(n_iterations),
		validation_data = validation_data,
		callbacks = [early_stopping]
	)
	p = model.predict_proba( x_train_, batch_size = params['batch_size'] )
	
	ll = log_loss( y_train, p )
	auc = AUC( y_train, p )
	acc = accuracy( y_train, np.round( p ))

	print ("\n# training | log loss: {:.2%}, AUC: {:.2%}, accuracy: {:.2%}".format( ll, auc, acc ))

	#

	p = model.predict_proba( x_test, batch_size = params['batch_size'] )

	ll = log_loss( y_test, p )
	auc = AUC( y_test, p )
	acc = accuracy( y_test, np.round( p ))

	print ("# testing  | log loss: {:.2%}, AUC: {:.2%}, accuracy: {:.2%}".format( ll, auc, acc ))

	return { 'loss': ll, 'log_loss': ll, 'auc': auc, 'early_stop': model.stop_training }




