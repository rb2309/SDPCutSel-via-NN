******Info on:**********
A. Package dependencies
B. How code is organized
C. Running provided code 

###############################################################################################
A. Dependencies
###############################################################################################

- OS: Linux/Windows x86-64
- Python 3.5 having the (non-standard packages):
	- CPLEX >=12.8 (Python API via IBM ILOG Optimization Suite)
	- MOSEK for Python (e.g. 'conda install -c mosek mosek')
	- ctypes, chompack, lxml, cvxopt, cvxpy=0.4.11 (version >=1.0 breaks current code),
	- (optional) Matlab + Coder + Neural/Deep Networks Toolbox for (re-)training neural nets used	

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
		- 'cut_select_powerflow.py' = cutting plane solver for powerflow (or more generally QCQP) instances
			- inherits from 'cut_select_qp.py' and adapts  Algorithm 1 to Section 5 
			- shown on an instance of powerflow0009r from MINLPlib (with pre-processed bounds)
	- 'utilities.py': implements data sampling (Section 4) for training and testing neural networks and
						the examples in Fig 1, 4-6
** Folders:
	- 'boxqp_instances' - all BoxQP instances (together with solutions) and the powerflow instance used
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

0. (IMPORTANT) Backup folders 'data_figures', 'data_tables' (Will get files overwritten!!!)
1. Run in terminal 'python generate_figs_tables', which has several configurations in its 'main()' function
(by default runs 'lighter' test configuration for data in all figures/tables):
a. Full vs 'lighter' test configuration
	- 'test_cfg'=False in 'main()' runs all BoxQP instances (named in 'boxqp_instances/filenames.txt'),
	some of which take very long to run (e.g. "spar125-075-1" for M+S_5 takes >>1h alone) and all figures fully
	- 'test_cfg'=True in 'main()' runs only small-medium BoxQP instances (named in 'boxqp_instances/filenames_test.txt')
	and reduced Figure 1 (reduced instance sizes to 30 max otherswise test takes ~10h) 
	and reduced Figure 9 (2 sparse instances only, otherwise exact selection takes very long ~5h) 
b. Different granularity (for each configuration) is provided in 'main()' (commented out):
	- Run everything, all figures, all tables, or one figure, or one table
2. To plot all figures run Matlab script 'plot_all' in "data_figures" folder
(to plot only figure X call 'figX_plot');
	- if figure is 5 or 7 (X=5,7) then call 'figX_plot(1)' to use newly sampled data 
	(not the data in the manuscript figures) if you want to see how neural nets do on new test data
	
********************************************************************************
How to recreate a particular table only (applies to tables 4-7, Figures 12-13)
********************************************************************************

For instance, to get Table X from manuscript (X in the table number):
1. call 'run_for_all_tables(X)' in generate_figs_tables.py / 'main()' 
	- optional parameters: 
		folder_name='data_tables' (folder to save raw .csv files and 'table_X.csv'in), 
		test_cfg=True(/False)
	- if X is 0, the run will generate data for ALL the tables

*******************************************************************************
How to recreate a particular figure only (applies to figs. 1, 4-6, 8-14)
*******************************************************************************

For instance, to get Figure X from manuscript (X in the figure number):
1. call 'run_for_figure(X)' in generate_figs_tables.py / 'main()' 
	- optional parameters: 
		folder_name='data_figures' (folder to save the 'figX_[...].csv' output data file in), 
		test_cfg=True(/False)
2. run matlab function 'figX_plot.m' from folder_name (default "data_figures") to visualize figure X;		
*** Exceptions: 
- Figures 10,11 data - call 'run_for_figure(10)' for both, but plot separately by running 'fig10_plot.m', 'fig11_plot.m';
- Figures 12-13 (require data from running all BoxQP instances, takes a while): 
	- call 'run_for_all_tables(table=12, test_cfg=False)' to save all the data needed only for figs 12-13 
	(raw data in folder 'data_tables' and aggregated data in 'data_figures/fig12_data.csv, fig13_data.csv');
	- call matlab script 'fig12_13_plot.m' for "data_figures" to plot;
	
*******************************************************************************
How to (re-)train neural networks for optimality cut sel. from scratch 
(Matlab Neural/Deep Networks Toolbox, Matlab Coder)
*******************************************************************************

1. Generate data for training in "neural_nets" folder, example: 
	'import utilities as utils
	save_file = os.path.join(dirname, "neural_nets", "data_3d.csv")
	utils.gen_data_ndim(10000, 3, save_file)'
1. Edit the first function in the Matlab script "neural_nets/train_NNs.m" to select:
	- which neural net to train (2 to 5 SDP dimensionality neural nets)
	- which data file to use and how many samples to use from it
	- warm-start from given checkpoint file or start from scratch
	- further customizations on how to train
2. Compile (Matlab Coder) trained Matlab neural net function with an existing .prj file or create one for your configuration
(load existing .prj files with Matlab Coder to check the input types/numbers for a neural net for a given SDP dimensionality)
3. Save NNs.[exec extension] (needs to have all neural nets for 2-5 dimensions compiled together in it) to "neural_nets" folder
4. The code uses the newly trained neural nets!
	
	
	
	
