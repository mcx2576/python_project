# Ethan Wipond
# v00773575
# A program that convert the output of minisat back into a solved sudoku puzzle


import sys, getopt

def main(argv):
	inputfile = ''
	outputfile = ''
	try: 
		opts, args = getopt.getopt(argv, 'hi:o:', ["ifile=", "ofile="])
	except getopt.GetoptError:
		print 'sudoku-to-cnf.py -i <inputfile> -o <outputfile>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'sudoku-to-cnf -i <inputfile> -o <outputfile>'
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-o", "--ofile"):
			outputfile = arg

	satisfying_assignment = read_input(inputfile)
	sudoku_string = convert_sat_to_sudoku(satisfying_assignment)
	write_output(outputfile, sudoku_string)


def read_input(filename):
	solved_cnf = ""
	with open(filename) as f:
		solved_cnf = f.read()
	list_of_vals = solved_cnf.split()

	return list_of_vals[1:]


def convert_sat_to_sudoku(satisfying_assignment):
	sudoku_board = []
	for index in satisfying_assignment:
		
		if int(index) > 0: # variable is true
			i = int(index) / 81
			index = int(index) % 81
			j = index / 9
			k = index % 9
			if(k == 0):
				k = 9
			sudoku_board.append(k) 
	return sudoku_board


def write_output(outputfile, string):
	o = open(outputfile, 'w')

	for i in range(9):
		for j in range(9):
			o.write("{0} ".format(string[9*i + j]))
		o.write("\n")
	o.close()



if __name__ == "__main__":
   main(sys.argv[1:])