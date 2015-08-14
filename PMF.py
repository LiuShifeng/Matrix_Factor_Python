#!/usr/bin/python
#
# Created by Shifeng Liu (2015)
#
# An implementation of matrix factorization
#
import os
import numpy
import math

class ProbabilisticMatrixFactorization():

    def __init__(self, rating_tuples, latent_d=1,beta=0.1,Wm=0.2):
        self.latent_d = latent_d
        self.learning_rate = .0001
        self.regularization_strength = beta
        self.Wm = Wm
        print Wm

        self.ratings = numpy.array(rating_tuples).astype(float)
        self.converged = False

        self.num_users = int(numpy.max(self.ratings[:, 0]) + 1)
        self.num_items = int(numpy.max(self.ratings[:, 1]) + 1)
        
        print (self.num_users, self.num_items, self.latent_d)
        print self.ratings

        self.users = numpy.random.random((self.num_users, self.latent_d))
        self.items = numpy.random.random((self.num_items, self.latent_d))

        self.new_users = numpy.random.random((self.num_users, self.latent_d))
        self.new_items = numpy.random.random((self.num_items, self.latent_d))           


    def loss(self, users=None, items=None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items
            
        sq_error = 0
        
        for rating_tuple in self.ratings:
            if len(rating_tuple) == 3:
                (i, j, rating) = rating_tuple
                weight = 1
            elif len(rating_tuple) == 4:
                (i, j, rating, weight) = rating_tuple
            
            r_hat = numpy.sum(users[i] * items[j])

            if rating == 0.0 :
                sq_error += self.Wm * weight * (rating - r_hat)**2
                print "Loss rating 0"
            else:
                sq_error += weight * (rating - r_hat)**2
        L2_norm = 0
        for i in range(self.num_users):
            for d in range(self.latent_d):
                L2_norm += users[i, d]**2

        for i in range(self.num_items):
            for d in range(self.latent_d):
                L2_norm += items[i, d]**2

        return sq_error + self.regularization_strength * L2_norm
        
        
    def update(self):

        updates_o = numpy.zeros((self.num_users, self.latent_d))
        updates_d = numpy.zeros((self.num_items, self.latent_d))        

        for rating_tuple in self.ratings:
            if len(rating_tuple) == 3:
                (i, j, rating) = rating_tuple
                weight = 1
            elif len(rating_tuple) == 4:
                (i, j, rating, weight) = rating_tuple
            
            r_hat = numpy.sum(self.users[i] * self.items[j])
            
            for d in range(self.latent_d):
                if rating == 0.0:
                    print "update rating 0 "
                    updates_o[i, d] += self.items[j, d] * (rating - r_hat) * weight * self.Wm
                    updates_d[j, d] += self.users[i, d] * (rating - r_hat) * weight * self.Wm
                else:
                    updates_o[i, d] += self.items[j, d] * (rating - r_hat) * weight
                    updates_d[j, d] += self.users[i, d] * (rating - r_hat) * weight

        while (not self.converged):
            initial_lik = self.loss()

            print "  setting learning rate =", self.learning_rate
            self.try_updates(updates_o, updates_d)

            final_lik = self.loss(self.new_users, self.new_items)

            if final_lik < initial_lik:
                self.apply_updates(updates_o, updates_d)
                self.learning_rate *= 1.25

                if initial_lik -  final_lik< 10:
                    self.converged = True
                    
                break
            else:
                self.learning_rate *= .5
                self.undo_updates()

            if self.learning_rate < 1e-10:
                self.converged = True

        return not self.converged
    

    def apply_updates(self, updates_o, updates_d):
        for i in range(self.num_users):
            for d in range(self.latent_d):
                self.users[i, d] = self.new_users[i, d]

        for i in range(self.num_items):
            for d in range(self.latent_d):
                self.items[i, d] = self.new_items[i, d]                

    
    def try_updates(self, updates_o, updates_d):        
        alpha = self.learning_rate
        beta = -self.regularization_strength

        for i in range(self.num_users):
            for d in range(self.latent_d):
                self.new_users[i,d] = self.users[i, d] + \
                                       alpha * (beta * self.users[i, d] + 2*updates_o[i, d])
        for i in range(self.num_items):
            for d in range(self.latent_d):
                self.new_items[i, d] = self.items[i, d] + \
                                       alpha * (beta * self.items[i, d] + 2*updates_d[i, d])
        

    def undo_updates(self):
        # Don't need to do anything here
        pass


    def print_latent_vectors(self):
        print "Users"
        for i in range(self.num_users):
            print i,
            for d in range(self.latent_d):
                print self.users[i, d],
            print
            
        print "Items"
        for i in range(self.num_items):
            print i,
            for d in range(self.latent_d):
                print self.items[i, d],
            print   
            
    def print_Users(self):
        print "Users"
        for i in range(self.num_users):
            print i,
            for d in range(self.latent_d):
                print self.users[i, d],
            print
            
    def print_Items(self):
        print "Items"
        for i in range(self.num_items):
            print i,
            for d in range(self.latent_d):
                print self.items[i, d],
            print    
            
    def save_Users(self,path):
         pwd=os.getcwd()
         outfile = open(os.path.join(pwd, path), 'w')
         for i in range(self.num_users):
            outfile.write (i)
            for d in range(self.latent_d):
                outfile.write(",")
                outfile.write (self.users[i, d])                
            outfile.write ('\n')
         outfile.close() 

    def save_Items(self,path):
         pwd=os.getcwd()
         outfile = open(os.path.join(pwd, path), 'w')
         for i in range(self.num_items):
            outfile.write (i)
            for d in range(self.latent_d):
                outfile.write(",")
                outfile.write (self.items[i, d])                
            outfile.write ('\n')
         outfile.close() 
         
    def save_latent_vectors(self, prefix):
        self.users.dump(prefix + "%sd_users.pickle" % self.latent_d)
        self.items.dump(prefix + "%sd_items.pickle" % self.latent_d)
    
    def rmse(self, users=None, items=None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items
            
        rmse = 0
        T = 0
        for rating_tuple in self.ratings:
            T = T +1
            if len(rating_tuple) == 3:
                (i, j, rating) = rating_tuple
                weight = 1
            elif len(rating_tuple) == 4:
                (i, j, rating, weight) = rating_tuple
            
            r_hat = numpy.sum(users[i] * items[j])

            if rating == float(0.0) :
                rmse += self.Wm * weight * (rating - r_hat)**2
            else:
                rmse += weight * (rating - r_hat)**2

        return math.sqrt(rmse/T)

    def topK_Hit_Ratio(self, users=None, items=None,K=5,relevent_bench=5):
        if users is None:
            users = self.users
        if items is None:
            items = self.items

        Hk = 0.0
        recall = 0.0
        Nu = [0 for i in range(self.num_users)]
        Nku = [0 for i in range(self.num_users)]
        sumNku = 0.0
        sumNu = 0.0

        for rating_tuple in self.ratings:
            if len(rating_tuple) == 3:
                (i, j, rating) = rating_tuple
                weight = 1
            elif len(rating_tuple) == 4:
                (i, j, rating, weight) = rating_tuple

            if rating >= relevent_bench:
                Nu[int(i)] += 1
                u = []
                for ii in range(self.num_items):
                    u.append(numpy.sum(users[int(i)] * items[ii]))
                u.sort(reverse = True)
                r_hat = numpy.sum(users[int(i)] * items[j])
                if u.index(r_hat) < K:
                    Nku[int(i)] += 1

        T = 0
        for i in range(self.num_users):
            if float(Nu[i]) > 0.0:
                T += 1
                sumNku += float(Nku[i])
                sumNu += float(Nu[i])
                Hk += Nku[i]/Nu[i]
        Hk = Hk/T
        recall = sumNku/sumNu
        return Hk,recall
        

def fake_ratings(noise=.25):
    u = []
    v = []
    ratings = []
    
    num_users = 100
    num_items = 100
    num_ratings = 30
    latent_dimension = 10
    
    # Generate the latent user and item vectors
    for i in range(num_users):
        u.append(2 * numpy.random.randn(latent_dimension))
    for i in range(num_items):
        v.append(2 * numpy.random.randn(latent_dimension))
        
    # Get num_ratings ratings per user.
    for i in range(num_users):
        items_rated = numpy.random.permutation(num_items)[:num_ratings]

        for jj in range(num_ratings):
            j = items_rated[jj]
            rating = numpy.sum(u[i] * v[j]) + noise * numpy.random.randn()
        
            ratings.append((i, j, rating))  # thanks sunquiang

    return (ratings, u, v)


def real_ratings(noise=.25):
    u = []
    v = []
    ratings = []
    
    num_users = 100
    num_items = 100
    latent_dimension = 10
    
    # Generate the latent user and item vectors
    for i in range(num_users):
        u.append(2 * numpy.random.randn(latent_dimension))
    for i in range(num_items):
        v.append(2 * numpy.random.randn(latent_dimension))
        
    # Get ratings per user.
    pwd=os.getcwd()
    infile = open(os.path.join(pwd, 'u.data'), 'r')
    for line in infile.readlines():
        f = line.rstrip('\r\n').split(",")
        f = (float(f[0]),float(f[1]),float(f[2]))
        ratings.append(f)

    return (ratings, u, v)

if __name__ == "__main__":

    #DATASET = 'fake'
    DATASET = 'real'

    if DATASET == 'fake':
        (ratings, true_o, true_d) = fake_ratings()
    if DATASET == 'real':
        (ratings, true_o, true_d) = real_ratings()
    

    #plot_ratings(ratings)

    pmf = ProbabilisticMatrixFactorization(ratings, latent_d=5, beta=0.01,Wm=0.0)
    
    liks = []
    print "before RMSE ",pmf.rmse()
    while (pmf.update()):
        lik = pmf.loss()
        liks.append(lik)
        print "L=", lik
        pass
    
    print "after RMSE ",pmf.rmse()
    Hk,recall = pmf.topK_Hit_Ratio()
    print Hk,recall
    '''
    pmf.save_Users('Mf\Users.data')
    pmf.save_Items('MF\Items.data')
    pmf.print_latent_vectors()
    
    pmf.save_latent_vectors("models/")
'''