#include <taskflow/taskflow.hpp>
#include "lavamd.hpp"								// (in the main program folder)	needed to recognized input variables

//	PLASMAKERNEL_GPU
void  taskflow_kernel(par_str par, 
	dim_str dim,
	box_str* box,
	FOUR_VECTOR* rv,
	fp* qv,
	FOUR_VECTOR* fv, unsigned num_threads) {

	//	MCPU SETUP
  tf::Executor executor(num_threads);
  tf::Taskflow taskflow;

	//	PROCESS INTERACTIONS
  taskflow.parallel_for(0, (int)dim.number_boxes, 1, [&par, dim, box, rv, qv, fv](int l){

    // home box
    long first_i;
    FOUR_VECTOR* rA;
    FOUR_VECTOR* fA;

    // parameters
    fp alpha;
    fp a2;

  	//	INPUTS
  	alpha = par.alpha;
  	a2 = 2.0*alpha*alpha;

    // neighbor box
    int pointer;
    long first_j; 
    FOUR_VECTOR* rB;
    fp* qB;

  	int i, j, k;

    // common
    fp r2; 
    fp u2;
    fp fs;
    fp vij;
    fp fxij,fyij,fzij;
    THREE_VECTOR d;


		//	home box - box parameters
		first_i = box[l].offset;												// offset to common arrays

		//	home box - distance, force, charge and type parameters from common arrays
		rA = &rv[first_i];
		fA = &fv[first_i];

		//	Do for the # of (home+neighbor) boxes
		for (k=0; k<(1+box[l].nn); k++) {
			//	neighbor box - get pointer to the right box
			if(k==0){
				pointer = l;													// set first box to be processed to home box
			}
			else{
				pointer = box[l].nei[k-1].number;							// remaining boxes are neighbor boxes
			}

			//	neighbor box - box parameters
			first_j = box[pointer].offset; 

			//	neighbor box - distance, force, charge and type parameters
			rB = &rv[first_j];
			qB = &qv[first_j];

			//	Do for the # of particles in home box
			for (i=0; i<NUMBER_PAR_PER_BOX; i=i+1){

				// do for the # of particles in current (home or neighbor) box
				for (j=0; j<NUMBER_PAR_PER_BOX; j=j+1){

					// // coefficients
					r2 = rA[i].v + rB[j].v - DOT(rA[i],rB[j]); 
					u2 = a2*r2;
					vij= exp(-u2);
					fs = 2.*vij;
					d.x = rA[i].x  - rB[j].x; 
					d.y = rA[i].y  - rB[j].y; 
					d.z = rA[i].z  - rB[j].z; 
					fxij=fs*d.x;
					fyij=fs*d.y;
					fzij=fs*d.z;

					// forces
					fA[i].v +=  qB[j]*vij;
					fA[i].x +=  qB[j]*fxij;
					fA[i].y +=  qB[j]*fyij;
					fA[i].z +=  qB[j]*fzij;

				} // for j

			} // for i

		} // for k

  }, num_threads);

  executor.run(taskflow).wait();
}

std::chrono::microseconds measure_time_taskflow(unsigned num_threads) {
  auto beg = std::chrono::high_resolution_clock::now();
  taskflow_kernel(par_cpu, dim_cpu, box_cpu,	rv_cpu,	qv_cpu,	fv_cpu, num_threads);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
}

