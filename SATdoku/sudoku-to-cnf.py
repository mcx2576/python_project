# Ethan Wipond
# v00773575

# A program that translates partially solved sudoku puzzles into 
# CNF formulas suitable for input to the minisat SAT solver

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

	build_clauses(inputfile, outputfile)


def build_clauses(inputfile, outputfile):	
	given_puzzle_string = read_input(inputfile);
	clauses = []	

	clauses += get_clauses_from_puzzle(given_puzzle_string)
	clauses += get_clauses_from_rule_1()
	clauses += get_clauses_from_rule_2()
	clauses += get_clauses_from_rule_3()
	clauses += get_clauses_from_rule_4()

	write_output(outputfile, clauses);


def read_input(filename):
	sudoku_puzzle = ""
	with open(filename) as f:
	  while True:
	    c = f.read(1)
	    if not c:
	      break
	    sudoku_puzzle += c
	sudoku_puzzle = ''.join(sudoku_puzzle.split()) 
	return sudoku_puzzle


def get_clauses_from_puzzle(puzzle_string):
	result = []
	for index in range(len(puzzle_string)):
		if puzzle_string[index] in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
			# puzzle_string[i] is a value given in the puzzle
			x = int(index / 9) + 1
			y = index % 9 + 1
			z = int(puzzle_string[index])
			variable = three_subscript_to_variable(x, y, z)
			result.append("{0} 0".format(variable))
	return result


def get_clauses_from_rule_1():
	# Every cell contains at least one number
	result = []
	for x in range(1,10):
		for y in range(1,10):
			this_clause = ""
			for z in range(1, 10):
				variable = three_subscript_to_variable(x, y, z)
				this_clause += "{0} ".format(variable) 
			this_clause += "0"
			result.append(this_clause)
	return result


def get_clauses_from_rule_2():
	# 2) Each number appears at most once in every row
	result = []
	for y in range(1,10):
		for z in range(1,10):
			for x in range(1,9):
				for i in range(x+1, 10):
					variable_1 = three_subscript_to_variable(x, y, z)
					variable_2 = three_subscript_to_variable(i, y, z)
					this_clause = "-{0} -{1} 0".format(variable_1, variable_2)
					result.append(this_clause)
	return result


def get_clauses_from_rule_3():
# 3) Each number appears at most once in every column
	result = []
	for x in range(1,10):
	 	for z in range(1,10):
	 		for y in range(1,9):
	 			for i in range(y+1, 10):
	 				variable_1 = three_subscript_to_variable(x, y, z)
	 				variable_2 = three_subscript_to_variable(x, i, z)
	 				this_clause = "-{0} -{1} 0".format(variable_1, variable_2)
	 				result.append(this_clause)
	return result


def get_clauses_from_rule_4():
# 4) Each number appears at most once in every 3x3 subgrid
	result = []
	for z in range(1,10):
		for i in range(0,3):
			for j in range(0,3):
				for x in range(1,4):
					for y in range(1,4):
						for k in range(y+1, 4):
							variable_1 = three_subscript_to_variable((3 * i + x), (3 * j + y), z)
							variable_2 = three_subscript_to_variable((3 * i + x), (3 * j + k), z)
							this_clause = "-{0} -{1} 0".format(variable_1, variable_2)
							result.append(this_clause)
	
	for z1 in range(1, 10):
		for i1 in range(0, 3):
			for j1 in range(0, 3):
				for x1 in range(1,4):
					for y1 in range(1,4):
						for k1 in range(x1+1, 4):
							for l1 in range(1, 4):
								variable_1 = three_subscript_to_variable((3 * i1 + x1), (3 * j1 + y1), z1)
								variable_2 = three_subscript_to_variable((3 * i1 + k1), (3 * j1 + l1), z1)
								this_clause = "-{0} -{1} 0".format(variable_1, variable_2)
								result.append(this_clause)
	return result



def write_output(outputfile, clauses):
	# format:
	# p cnf <# variables> <# clauses>
	# <list of clauses>
	# each clause is a list of non-zero numbers terminated by a 0
	o = open(outputfile, 'w')

	num_variables = 729
	num_clauses = len(clauses)
	o.write("p cnf {0} {1}\n".format(num_variables, num_clauses))

	for i in range(len(clauses)):
		o.write("{0}\n".format(clauses[i]))
	o.close()


def three_subscript_to_variable(x, y, z):
	return 81 * (x-1) + 9 * (y-1) + (z-1) + 1


if __name__ == "__main__":
   main(sys.argv[1:])