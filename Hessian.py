import os
from pyspark import SparkConf, SparkContext
import math
import sys
import timeit
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry,RowMatrix,IndexedRowMatrix
import argparse
import numpy as np
import psutil
import sys
from operator import add
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext  

if __name__ == "__main__":
	t0 = timeit.default_timer()


	parser = argparse.ArgumentParser()
	parser.add_argument('pdb', type=str, help='Path to Input pdb file')
	parser.add_argument('cutoff',type=int, help='Cutoff')
	parser.add_argument('gamma',type=int, help='gamma')
	parser.add_argument('k',type=int, help='Number of singular modes')
	args = parser.parse_args()

	conf = SparkConf().setAppName("KirchhoffApp")
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)  

	textFile = sc.textFile(args.pdb);
	pairs=textFile.map(lambda line: (line.split()[0],line));
	CAATOMlines = pairs.filter(lambda (key, value): value.split()[0]=="ATOM" and value.split()[2]=="CA");
	chain_coor = CAATOMlines.map(lambda (key, value): (str(value)[21:22], float(str(value)[30:38]), float(str(value)[39:46]), float(str(value)[47:54]) ));
	coors = CAATOMlines.map(lambda (key, value): (float(str(value)[30:38]), float(str(value)[39:46]), float(str(value)[47:54]) ));
	
	Chains_Natoms=CAATOMlines.map(lambda (key, value): (str(value)[21:22],1)).reduceByKey(lambda x, y: x + y).sortByKey();
	Chains = Chains_Natoms.map(lambda line: line[0]);
	Natoms = Chains_Natoms.map(lambda line: line[1]);


	coo_matrix_input = sc.emptyRDD()


	Total_atoms = 0
	cutoff=args.cutoff
	gamma=args.gamma
	mb = 1024*1024


	#Build Hessian
	Coors_ind = coors.zipWithIndex().map(lambda (values, key): (key,values));
	Combs=Coors_ind.cartesian(Coors_ind).filter(lambda (x,y): x[0] < y[0]);
	joinReady_Combs = Combs.map(lambda (x,y): ((x[0], y[0]), (x[1],y[1])));

	#Compute RMSD
	rdd_cartesian_O=Combs.map(lambda ((id1, (x1, y1,z1)), (id2, (x2, y2,z2))): ((id1, id2), (x1 - x2)*(x1-x2) + (y1 - y2)*(y1-y2)+(z1-z2)*(z1-z2)));
	rdd2=rdd_cartesian_O.filter(lambda (x,y): y < cutoff*cutoff);
	rdd3= rdd2.filter(lambda (x,y): y!= 0);

	Combs_rel = joinReady_Combs.join(rdd3).mapValues(lambda x: x[0]);
	Combs_rmsd = Combs_rel.join(rdd3);
	Combs_rmsd.cache();

	rdd_cartesian = Combs_rmsd.map(lambda (x,y): (3*x[0]+0, 3*x[1]+0,(y[0][1][0]-y[0][0][0])*(y[0][1][0]-y[0][0][0])/(y[1]))).union(Combs_rmsd.map(lambda (x,y): (3*x[0]+0, 3*x[1]+1,(y[0][1][2]-y[0][0][0])*(y[0][1][1]-y[0][0][1])/(y[1])))).union(Combs_rmsd.map(lambda (x,y): (3*x[0]+0, 3*x[1]+2,(y[0][1][0]-y[0][0][0])*(y[0][1][2]-y[0][0][2])/(y[1])))).union(Combs_rmsd.map(lambda (x,y): (3*x[0]+1, 3*x[1]+0,(y[0][1][1]-y[0][0][1])*(y[0][1][0]-y[0][0][0])/(y[1])))).union(Combs_rmsd.map(lambda (x,y): (3*x[0]+1, 3*x[1]+1,(y[0][1][1]-y[0][0][1])*(y[0][1][1]-y[0][0][1])/(y[1])))).union(Combs_rmsd.map(lambda (x,y): (3*x[0]+1, 3*x[1]+2,(y[0][1][1]-y[0][0][1])*(y[0][1][2]-y[0][0][2])/(y[1])))).union(Combs_rmsd.map(lambda (x,y): (3*x[0]+2, 3*x[1]+0,(y[0][1][2]-y[0][0][2])*(y[0][1][0]-y[0][0][0])/(y[1])))).union(Combs_rmsd.map(lambda (x,y): (3*x[0]+2, 3*x[1]+1,(y[0][1][2]-y[0][0][2])*(y[0][1][1]-y[0][0][1])/(y[1])))).union(Combs_rmsd.map(lambda (x,y): (3*x[0]+2, 3*x[1]+2,(y[0][1][2]-y[0][0][2])*(y[0][1][2]-y[0][0][2])/(y[1]))));
	coo_matrix_input = coo_matrix_input.union(rdd_cartesian)

	# Transpose the matrix
	coo_matrix_input_LT = coo_matrix_input.map( lambda (i,j,k): (j,i,k));
	coo_matrix_input_all = coo_matrix_input_LT.union(coo_matrix_input);
	coo_matrix_input_all.cache()


	# Diagonalize RDD  

	diag_entries_1 = coo_matrix_input_all.filter(lambda (row, col, value): col%3 ==0).map(lambda (row, _, value): (row, value)).reduceByKey(lambda x, y: x + y).map(lambda (row,value): (row, 3*(row/3),-value ));
	diag_entries_1.cache()
	diag_entries_2 = coo_matrix_input_all.filter(lambda (row, col, value): col%3 ==1).map(lambda (row, _, value): (row, value)).reduceByKey(lambda x, y: x + y).map(lambda (row,value): (row, 3*(row/3)+1,-value ));
	diag_entries_2.cache()
	diag_entries_3 = coo_matrix_input_all.filter(lambda (row, col, value): col%3 ==2).map(lambda (row, _, value): (row, value)).reduceByKey(lambda x, y: x + y).map(lambda (row,value): (row, 3*(row/3)+2,-value ));
	diag_entries_3.cache()

	diag_entries = diag_entries_1.union(diag_entries_2).union(diag_entries_3);
	
	coo_matrix_input_all  = coo_matrix_input_all.union(diag_entries);
	coo_matrix_entries = coo_matrix_input_all.map(lambda e: MatrixEntry(e[0], e[1], e[2]));
	coo_matrix = CoordinateMatrix(coo_matrix_entries);


	#SAVE TO A FILE
	coo_matrix_input_all.repartition(1).saveAsTextFile("./Laplacian_4v7o_4cores_1")
	t2 = timeit.default_timer()
	print("Elapsed time for construction: {:} s".format(t2 - t0))


	#Singular value decomposition
	
	dataRows = coo_matrix.toRowMatrix().rows

	k = int(args.k) #N_singvalues
	svd = RowMatrix(dataRows.persist()).computeSVD(k, computeU=True)
	U = svd.U # The U factor is a RowMatrix.
	s = svd.s # The singular values are stored in a local dense vector.
	V = svd.V #The V factor is a local dense matrix


	sc.parallelize(V.toArray()).repartition(1).saveAsTextFile("EigenVectors_4v7o_4cores")
	sc.parallelize(s.toArray()).repartition(1).saveAsTextFile("EigenValues_4v7o_4cores")

	t4 = timeit.default_timer()
	print("Elapsed time for SVD: {:} s".format(t4 - t2))
	print("Total memory = {:}, used memory = {:}, free memory = {:}".format(psutil.virtual_memory().total/mb, (psutil.virtual_memory().total - psutil.virtual_memory().free) / mb, psutil.virtual_memory().free/mb));
	print("System size = {:} atoms".format(Natoms.sum()))
	print("No. of chains ={:}".format(Chains.count()))






