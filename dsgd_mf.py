import sys, random, numpy
from numpy.random import rand
from numpy import savetxt
from pyspark import SparkContext, SparkConf


def main(num_factors, num_workers, num_iterations, beta_value, lambda_value, Wm_value, \
        V_filename, output_W_filename, output_H_filename):
    # Conf
    conf = SparkConf().setAppName("Spark SGD MF")
    sc = SparkContext(conf=conf)
    
    user_movie_ratings = sc.textFile(V_filename).map(line_to_movie_user_ratings)
    user_movie_ratings.persist()

    #global user_nonzero, movie_nonzero
    #user_nonzero = user_movie_ratings.keyBy(first_element).countByKey()
    #movie_nonzero = user_movie_ratings.keyBy(second_element).countByKey()

    num_users = int(user_movie_ratings.map(first_element).reduce(max))
    num_movies = int(user_movie_ratings.map(second_element).reduce(max))

    global updates_total
    updates_total = 0
   
    # Begin iterations
    iter = 0
    global seed
    while iter < num_iterations:
        # Initialize W and H
        if iter == 0:
            W = sc.parallelize(range(num_users+1)).map(key_to_entry_rand).persist()#(user_id,rand(num_factors))
            H = sc.parallelize(range(num_movies+1)).map(key_to_entry_rand).persist()#(movie_id,rand(num_factors)

        # Set random seed
        seed = random.randrange(MAXSEED)

        # Partition parameters
        W_blocks = W.keyBy(lambda W_entry: item_to_block(W_entry[0]))#key:worker_id,value:(user_id,rand(num_factors))
        H_blocks = H.keyBy(lambda H_entry: item_to_block(H_entry[0]))#key:worker_id,value:(movie_id,rand(num_factors)

        # Filter diagonal blocks
        V_diagonal = user_movie_ratings.filter(filter_diagonal).persist()#(user_id,movie_id,rating) where worker_id(user_id) == worker_id(movie_id)
        V_blocks = V_diagonal.keyBy(lambda t : item_to_block(t[0]))#key:worker_id,value:(user_id,movie_id,rating) where user_id == movie_id
        updates_curr = V_diagonal.count()
        V_diagonal.unpersist()    
        V_group = V_blocks.groupWith(W_blocks, H_blocks).coalesce(num_workers)#key:worker_id,value:seq[V],seq[W],seq[H]

        # Perform SGD
        updatedWH = V_group.map(SGD_update).persist()
        W = updatedWH.flatMap(first_element).persist()
        H = updatedWH.flatMap(second_element).persist()
        updates_total += updates_curr
        iter += 1
   
    W_result = numpy.vstack(W.sortByKey().map(second_element).collect()[1:])
    H_result = numpy.vstack(H.sortByKey().map(second_element).collect()[1:])
    # Save W and H
    savetxt(output_W_filename, W_result, delimiter=',')
    savetxt(output_H_filename, H_result, delimiter=',')
    sc.stop


# id to block id
def item_to_block(item_idx):
    return hash(str(item_idx) + str(seed)) % num_workers

def line_to_movie_user_ratings(line):
    user_movie_rating = [float(item) for item in line.strip().split(',')]
    user_movie_rating[0] = int(user_movie_rating[0])
    user_movie_rating[1] = int(user_movie_rating[1])
    return user_movie_rating

def key_to_entry_rand(t):
    return (t, rand(num_factors))

def filter_diagonal(t):
    return item_to_block(t[0]) == item_to_block(t[1])

def first_element(t):
    return t[0]

def second_element(t):
    return t[1]

# SGD update W and H
def SGD_update(t):
    V_block, W_block, H_block = t[1]
    W_dict = dict(W_block)
    H_dict = dict(H_block)
    iter = 0
    for (movie_id, user_id, rating) in V_block:
        iter += 1
        epsilon = pow(100+updates_total+iter, -1*beta_value)  
        Wi = W_dict[int(movie_id)]
        Hj = H_dict[int(user_id)]
        # loss
        loss = -2*(rating - numpy.dot(Wi,Hj))
        if rating == 0.0:
            loss = Wm_value * loss
        H_dict[int(user_id)]  = Hj - epsilon*(2*lambda_value*Hj + loss*Wi)
        W_dict[int(movie_id)] = Wi - epsilon*(2*lambda_value*Wi + loss*Hj)
    return (W_dict.items(), H_dict.items())


if __name__ == "__main__":
    if len(sys.argv) <> 10:
        print ''
        print 'Usage: spark-submit %s <num_factors> <num_workers> \
                <num_iterations> <beta_value> <lambda_value> <Wm_value> <V_filename> \
                <output_W_filename> <output_H_filename>' % sys.argv[0]
        print ''
        sys.exit(1)
    
    # Set global vars
    global num_factors, num_workers, num_iterations, \
            beta_value, lambda_value, Wm_value, \
            V_filename, output_W_filename, output_H_filename
    num_factors, num_workers, num_iterations = map(int, sys.argv[1:4])
    beta_value, lambda_value, Wm_value = map(float, sys.argv[4:7])
    V_filename, output_W_filename, output_H_filename = sys.argv[7:10]
    
    # Random max seed
    global MAXSEED
    MAXSEED = 100000
    # Start map reduce
    main(num_factors, num_workers, num_iterations, beta_value, lambda_value, Wm_value, \
            V_filename, output_W_filename, output_H_filename)

