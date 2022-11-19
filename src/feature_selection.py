from itertools import count
from numpy.random import randint
from numpy.random import rand
from model import evaluate_model
 

def fitness(chromosome, n_bits):
    score, number_of_features = evaluate_model(chromosome)
    score_weight = 0.8
    number_of_features_weight = 0.2
    return (score_weight * score) - (number_of_features_weight * (number_of_features / n_bits))
 

# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop)) 
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] > scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]
 

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]
 

# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]
 

# genetic algorithm
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
	# initial population of random bitstring
	pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]

	# keep track of best solution
	best, best_eval, best_gen = 0, 0, 0
	# enumerate generations
	for gen in range(n_iter):
		print(f'\n\nGeneration: {gen}')
		# evaluate all candidates in the population
		print('\n\nCalculating fitness for current population...')
		scores = [objective(c, n_bits) for c in pop]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] > best_eval:
				best, best_eval, best_gen = pop[i], scores[i], gen
				print("\n\n>generation: %d -> new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
		# select parents
		print('\n\nSelection...')
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		print('\n\nCrossover and mutation...')
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
	return [best, best_eval, best_gen]
 

# define the total iterations
n_iter = 50
# bits
n_bits = 69
# define the population size (must be even!)
n_pop = 16
# crossover rate
r_cross = 0.7
# mutation rate
r_mut = 1.0 / float(n_bits)

# perform the genetic algorithm search
best, score, gen = genetic_algorithm(fitness, n_bits, n_iter, n_pop, r_cross, r_mut)
print('\n\nResult:\n')
print(f'generation {gen} -> f({best}) = {score}')

#save result

#open text file
text_file = open("./feature-selection-result.txt", "w")
 
#write string to file
text_file.write(f'generation {gen} -> f({best}) = {score}')
 
#close file
text_file.close()
