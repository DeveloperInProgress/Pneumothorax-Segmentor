import _pickle as pickle

data_file = 'data/pneumodata.pckl'

print ("loading...")

with open( data_file, 'rb' ) as f:
	data = pickle.load( f )
