******Info on:**********
A. Package dependencies
B. How code is organized
C. Running provided code 

###############################################################################################
A. Dependencies
###############################################################################################

- OS: Linux/Windows x86-64
- Python 3.5 having the (non-standard packages):
	- CPLEX 12.8 (Python API via IBM ILOG Optimization Suite)
	- MOSEK for Python - 'conda install -c mosek mosek'
	- cython, chompack, lxml, cvxopt=1.2, cvxpy=0.4.11 (version >=1.0 breaks current unchanged code) 
	- numpy==1.14.2, scipy==1.1.0 (to get the same seed behaviour on scipy.stats.ortho_group/ the same random results)
	Note any version change, particularly for numpy/scipy, will change numbers very slightly due to numerical sensitivity
	e.g. final bounds +- <0.1% due to convergence tolerance used of 0.1%.
- (optional) Matlab and 
	- YALMIP + SeDuMi for running 'test_sedumi.m' for Table 2 	
	- Coder + Neural/Deep Networks Toolbox for (re-)training neural nets used
	
** Python setup (performed):
	- conda virtual environment 'py35':
		cd <folder>
		curl -O https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
		bash Anaconda3-2018.12-Linux-x86_64.sh
			- yes to default <folder>/anaconda3 path and to prepend path to PATH
		conda create -n py35_used python=3.5.6
		conda activate py35_used
		conda install -c mosek mosek (and added academic license file <folder>/mosek/mosek.lic)
		conda install -c anaconda cvxopt
		conda install -c sebp cvxpy=0.4.11
		pip install --upgrade pip (or python -m pip install --upgrade pip)
		pip install -r <folder>/mpc_code/requirements.txt
	- Installed CPLEX 12.8 locally in <folder>
		cd <folder>/cplex/python/3.6/x86-64_linux/
		python setup.py build -b <folder>/build/ install
** Optional
	- Install Matlab (for plotting) and Coder + Neural/Deep Networks Toolbox
	- Install SeDuMi:
		- download SeDumi 1.3 from http://sedumi.ie.lehigh.edu/?page_id=58 
		- Add folder to MATLAB path and if binaries don't work follow Install.txt and recompile it
	- Install YALMIP (to call SeDuMi and set some parameters):
		- follow instructions at https://yalmip.github.io/tutorial/installation/
		- Add folder to MATLAB path 

###############################################################################################
B. How code is organized
###############################################################################################

** Python files (all well documented)):
	- 'generate_figs_tables.py' = entry point through its main function
		- used to generate the data for all manuscript tables and figures
		- instantiates the two classes below
	- 2 files with solver classes:
		- 'cut_select_qp.py' = cutting plane solver for QP
			- implements Algorithm 1/Sections 2-5 on BoxQP instances
		- 'cut_select_qcqp.py' = cutting plane solver for box constrained QCQP instances
			- inherits from 'cut_select_qp.py' and adapts Algorithm 1 to Section 6 
	- 'utilities.py': implements data sampling (Section 4) for training and testing neural networks and
						the examples in Fig 1, 3-6 and times reported in Table 2 for NN/Mosek
** Matlab files - 'test_sedumi.m' prints timings for running SeDuMi with different tolerances as reported in Table 2 						
** Folders:
	- 'boxqp_instances' - all BoxQP instances (together with solutions) and the powerflow instance used
	- 'boxqp_instances' - the box-constrained QCQP instance set 'qcqp3' from https://www.minlp.com/nlp-and-minlp-test-problems ,
						converted to osil format, and 'qcqp_sols.txt' lists feasible solutions found via GAMS/CONOPT solver
	- 'data_figures' - default location for all figures data:
						all .csv data files, their corresponding plots (.fig and .png) and the Matlab scripts
						to convert the data files to plots
	- 'data_tables' - default location for all tables data:
						all .csv data files with info for each BoxQP instance solved (raw file) and
						aggregated .csv files corresponding to the manuscript tables
	- 'neural_nets' - the trained neural nets used in the solvers/manuscript results as Matlab functions (2-5 dimensional) 
					and compiled Win/Linux C libraries (NNs.dll/so), the Matlab Coder projects to compile them 
					and Matlab checkpoint files to continue training them (from current parameters on any data), 
					and training script 'train_NNs.m' (well-documented) which can use existing/new data to continue/start training;
	
###############################################################################################
C. How to run code - easy/customizable run for all figures/tables in manuscript 
###############################################################################################

0. (IMPORTANT) Backup folders 'data_figures', 'data_tables' (will get files overwritten!) or modify folders in 'generate_figs_tables.py/main()'
	- Run 'conda activate py35' (to be in right Python environment)
1. Run 'python generate_figs_tables.py', which has several configurations in its 'main()' function
(by default runs 'lighter' test configuration for data in all figures/tables):
a. Full vs 'lighter' test configuration
	- 'test_cfg'=False in 'main()' runs all BoxQP instances (named in 'boxqp_instances/filenames.txt'),
	some of which take very long to run (e.g. 'spar125-075-1' for M+S_5 takes >>1h alone with dual simplex) and all figures fully
	- 'test_cfg'=True in 'main()' runs only small-medium BoxQP instances (named in 'boxqp_instances/filenames_test.txt')
	and reduced Figure 1 (reduced instance sizes to 30 max otherswise test takes ~10h) 
	and reduced Figures 9/10 (1 sparse instance only, otherwise exact selection takes very long ~5h) 
b. Different granularity (for each configuration) is provided in 'main()' (commented out):
	- Run everything, all figures, all tables, or one figure, or one table
2. To plot all figures run Matlab script 'plot_all' in "data_figures" folder
(to plot only figure X call 'figX_plot');
	- if figure is 4 or 7 (X=4,7) - the data will be saved in different .csv files in 'neural_nets' folder, keeping the originals 
	which will be plotted by default - to plot new data see instructions below
		
********************************************************************************
How to recreate a particular table only (applies to tables 4-7, Figures 12-13)
********************************************************************************

For instance, to get Table X from manuscript (X in the table number):
1. call 'run_for_all_tables(X)' in 'generate_figs_tables.py/main()' 
	- optional parameters: 
		folder_name='data_tables' (folder to save raw .csv files and 'table_X.csv'in), 
		test_cfg=True(/False)
	- if X is 0, the run will generate data for ALL the tables

*******************************************************************************
How to recreate a particular figure only (applies to figs. 1, 4-6, 8-14)
*******************************************************************************

For instance, to get Figure X from manuscript (X in the figure number):
1. call 'run_for_figure(X)' in 'generate_figs_tables.py/main()' 
	- optional parameters: 
		folder_name='data_figures' (folder to save the 'figX_[...].csv' output data file in), 
		test_cfg=True(/False)
2. run matlab function 'figX_plot.m' from folder_name (default "data_figures") to visualize figure X;		
*** Exceptions: 
- Figures 10,11 data - call 'run_for_figure(10)' for both, but plot separately by running 'fig10_plot.m', 'fig11_plot.m';
- Figures 12-13 (require data from running all BoxQP instances, takes a while): 
	- call 'run_for_all_tables(table=12)' to save all the data needed only for figs 12-13 
	(raw data in folder 'data_tables' and aggregated data in 'data_figures/fig12_data.csv, fig13_data.csv');
	- call matlab script 'data_figures/fig12_13_plot.m' to plot;
- Figures 4, 7 - the data will be saved in different .csv files in 'neural_nets' folder (keeping the originals)
	- Can check if the first data points match the existing data (same seed=7 used so should be the generating same data)
	- call Matlab functions 'fig4_plot()','fig7_plot()', to plot originals from manuscript (default in 'plot_all')
	or simply pass any parameter e.g. 'fig4_plot(1)','fig7_plot(1)' to plot based on the new .csv files (new sampled data)	
	
*******************************************************************************
How to (re-)train neural networks for optimality cut sel. from scratch 
(Matlab Neural/Deep Networks Toolbox, Matlab Coder)
*******************************************************************************

1. Generate data for training in 'neural_nets' folder, example: 
	'import utilities as utils
	save_file = os.path.join(dirname, "neural_nets", "data_3d.csv")
	utils.gen_data_ndim(10000, 3, save_file)'
1. Edit the first function in the Matlab script 'neural_nets/train_NNs.m' to select:
	- which neural net to train (2 to 5 SDP dimensionality neural nets)
	- which data file to use and how many samples to use from it
	- warm-start from given checkpoint file or start from scratch
	- further customizations on how to train
2. Compile (Matlab Coder) trained Matlab neural net function with an existing .prj file or create one for your configuration
(load existing .prj files with Matlab Coder to check the input types/numbers for a neural net for a given SDP dimensionality)
3. Save NNs.[exec extension] (needs to have all neural nets for 2-5 dimensions compiled together in it) to "neural_nets" folder
4. The code uses the newly trained neural nets!