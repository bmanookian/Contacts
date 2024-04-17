import numpy as np
import sys
sys.path.append('/home/bmanookian/Contacts/')
import getContactProcess as gC

# INPUT FILES
  # if using a get Contacts tsv file put directory below 
inputtsv=None

  # if using a csv traj file put fil directory below
csvtraj='/data/nts/traj.csv'

  # numpy traj file - for numpy trajectory include both the trajectory and labels numpys
numpytraj=None
numpylabels=None

# PARAMETERS
# remove singles and neighbors
removesingles=True
removeneighbors=True;neighbors=1

# compute the MI matrix - if MI given provide numpy
computeMImarix=True;numprocs=28;MIfile='/directory/to/MI/file.npy'

# Use MI to remove contacts and save new csv
findremovepairs=True;thresh=0.01
outputcsv=inputtsv[:-4]+f'_th{thresh}.csv'
MIgiven=False

# Extract input trajectory from contact file
if inputtsv is not None:
	md=gC.traj_from_contact(fin=inputtsv)

	print('Input number of features:',md.unqpair.shape[0])
	np.save(inputtsv[:-4]+'_traj.npy',md.traj)
	np.save(inputtsv[:-4]+'_unqpair.npy',md.unqpair)

if csvtraj is not None:
	csvdata=gC.datareader(csvtraj)
	md=gC.traj_from_contact(traj=csvdata[1:,:].astype(int),unqpair=csvdata[0,:])
if numpytraj is not None:
	md=gC.traj_from_contact(traj=np.load(numpytraj),unqpair=np.load(numpylabels))
	print('Input number of features:',md.unqpair.shape[0])


# Remove singles and Neighbors if needed
if removesingles==True:
	md.remove_singles()
if removeneighbors==True:
	md.remove_Neighbor(neighbors)

print('Number of features after first round of feature selection:',md.unqpair.shape[0])


# Compute MI matrix and find low MI features to remove

if computeMImarix==True:
	if MIgiven==True:
		md.compute_MI_matrix(numprocs,MI=np.load(MIfile))
	else:
		md.compute_MI_matrix(numprocs)
		np.save(inputtsv[:-4]+'_MI.npy',md.MI)

if findremovepairs==True:
	md.find_pairs_to_remove(thresh)

print('Number of features after removing low MI features:',md.unqpair.shape[0])
	

# Write to output

print('Trajectory written to output with shape:',md.traj.shape)

gC.datawrite(outputcsv,md.traj.astype(int),labels=md.pairValid)




