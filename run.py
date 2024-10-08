import numpy as np
import sys
sys.path.append('/home/bmanookian/Contacts/')
import getContactProcess as gC

# outputdir
outdir='./'

# INPUT FILES
  # if using a get Contacts tsv file put directory below 
inputtsv='./180ps.tsv'

  # if using a csv traj file put fil directory below
csvtraj=None

  # numpy traj file - for numpy trajectory include both the trajectory and labels numpys
numpytraj=None#'traj.npy'
numpylabels=None#'labels.npy'

# PARAMETERS
#Cu Trajectory
cuttraj=False
start=45000;end=63352

# remove singles and neighbors
removesingles=True
removeneighbors=True;neighbors=1

# compute the MI matrix - if MI given provide numpy
MIfile=None#'MI.npy'
computeMImarix=True;numprocs=28

# Use MI to remove contacts and save new csv
findremovepairs=True;thresh=0.005
outputcsv=outdir+f'th{thresh}.csv'


# Extract input trajectory from contact file
if inputtsv is not None:
	md=gC.traj_from_contact(fin=inputtsv)

	print('Input number of features:',md.unqpair.shape[0])
	np.save(outdir+'traj.npy',md.traj)
	np.save(outdir+'labels.npy',md.unqpair)

if csvtraj is not None:
	csvdata=gC.datareader(csvtraj)
	md=gC.traj_from_contact(traj=csvdata[1:,:].astype(int).T,unqpair=csvdata[0,:])
	print('Input number of features:',md.unqpair.shape[0])
if numpytraj is not None:
	md=gC.traj_from_contact(traj=np.load(numpytraj),unqpair=np.load(numpylabels))
	print('Input number of features:',md.unqpair.shape[0])



# Cut traj
if cuttraj==True:
	md.cuttraj(start,end)


# Remove singles and Neighbors if needed
if removesingles==True:
	md.remove_singles()
if removeneighbors==True:
	md.remove_Neighbor(neighbors)

print('Number of features after first round of feature selection:',md.unqpair.shape[0])


# Compute MI matrix and find low MI features to remove

if computeMImarix==True:
	if MIfile is not None:
		md.compute_MI_matrix(numprocs,MI=np.load(MIfile))
	else:
		md.compute_MI_matrix(numprocs)
		np.save(outdir+f'MI.npy',md.MI)

if findremovepairs==True:
	md.find_pairs_to_remove(thresh)

print('Number of features after removing low MI features:',md.unqpair.shape[0])
	

# Write to output

print('Trajectory written to output with shape:',md.traj.T.shape)

gC.datawrite(outputcsv,md.traj.astype(int).T,labels=md.unqpair)




