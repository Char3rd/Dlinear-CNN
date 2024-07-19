#!/usr/bin/env python

"a more polished example of using hyperband"
"includes displaying best results and saving to a file"

import sys
import pickle 
from pprint import pprint

from hyperband import Hyperband

from defs import get_params, try_params

try:
	output_file = sys.argv[1]
	if not output_file.endswith( '.pkl' ):
		output_file += '.pkl'	
except IndexError:
	output_file = 'results.pkl'
	
print ("Will save results to", output_file)

#

hb = Hyperband( get_params, try_params )
results = hb.run()

print ("{} total, best:\n".format( len( results )))

for r in sorted( results, key = lambda x: x['loss'] )[:5]:
	print ("loss: {:.2%} | {} seconds | {:.1f} iterations | run {} ".format( 
		r['loss'], r['seconds'], r['iterations'], r['counter'] ))
	try:
		print (f"mse: {r['mse']}\tmae: {r['mae']}")
	except:
		pass
	pprint( r['params'] )

print ("saving...")

with open( output_file, 'wb' ) as f:
	pickle.dump( results, f )