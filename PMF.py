#!/usr/bin/python
#
# Created by Shifeng Liu (2015)
#
# An implementation of matrix factorization
#
import os
import sys
import numpy as np
import math

class ProbabilisticMatrixFactorization():

    def __init__(self, rating_tuples, latent_d=1,beta=0.1,Wm=0.2):
        self.latent_d = latent_d
        self.learning_rate = .0001
        self.regularization_strength = beta
        self.Wm = Wm
        print Wm

        self.ratings = np.array(rating_tuples).astype(float)
        self.converged = False

        self.num_users = int(np.max(self.ratings[:, 0]) + 1)
        self.num_items = int(np.max(self.ratings[:, 1]) + 1)
        
        print (self.num_users, self.num_items, self.latent_d)
        print self.ratings

        self.users = np.random.random((self.num_users, self.latent_d))
        self.items = np.random.random((self.num_items, self.latent_d))

        self.new_users = np.random.random((self.num_users, self.latent_d))
        self.new_items = np.random.random((self.num_items, self.latent_d))

        self.current_loss = self.loss()

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
            
            r_hat = np.sum(users[i] * items[j])

            if rating == 0.0 :
                sq_error += self.Wm * weight * (rating - r_hat)**2
            else:
                sq_error += weight * (rating - r_hat)**2
        L2_norm = 0
        L2_norm += np.sum(users*users)
        L2_norm += np.sum(items*items)

        return sq_error + self.regularization_strength * L2_norm

    def update(self):

        updates_o = np.zeros((self.num_users, self.latent_d))
        updates_d = np.zeros((self.num_items, self.latent_d))

        for rating_tuple in self.ratings:
            if len(rating_tuple) == 3:
                (i, j, rating) = rating_tuple
                weight = 1
            elif len(rating_tuple) == 4:
                (i, j, rating, weight) = rating_tuple
            
            r_hat = np.sum(self.users[i] * self.items[j])

            if rating == 0.0:
                updates_o[i] += self.items[j] * (rating - r_hat) * weight * self.Wm
                updates_d[j] += self.users[i] * (rating - r_hat) * weight * self.Wm
            else:
                updates_o[i] += self.items[j] * (rating - r_hat) * weight
                updates_d[j] += self.users[i] * (rating - r_hat) * weight

        while (not self.converged):
            initial_lik = self.current_loss

            print "  setting learning rate =", self.learning_rate
            self.try_updates(updates_o, updates_d)

            final_lik = self.loss(self.new_users, self.new_items)

            if final_lik < initial_lik:
                self.apply_updates(updates_o, updates_d)
                self.learning_rate *= 1.25

                if initial_lik -  final_lik< 10:
                    self.converged = True
                self.current_loss = final_lik
                break
            else:
                self.learning_rate *= .5
                self.undo_updates()

            if self.learning_rate < 1e-10:
                self.converged = True

        return not self.converged

    def apply_updates(self, updates_o, updates_d):
        self.users = np.copy(self.new_users)
        self.items = np.copy(self.new_items)
    
    def try_updates(self, updates_o, updates_d):        
        alpha = self.learning_rate
        beta = -self.regularization_strength

        for i in range(self.num_users):
            self.new_users[i] = self.users[i] + \
                                       alpha * (beta * self.users[i] + 2*updates_o[i])
        for i in range(self.num_items):
            self.new_items[i] = self.items[i] + \
                                       alpha * (beta * self.items[i] + 2*updates_d[i])
        

    def undo_updates(self):
        # Don't need to do anything here
        pass

    def get_current_loss(self):
        return self.current_loss

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
         outfile = open(path, 'w')
         for i in range(self.num_users):
            outfile.write (str(self.users[i, 0]))
            for d in range(1,self.latent_d):
                outfile.write(",")
                outfile.write (str(self.users[i, d]))
            outfile.write ('\n')
         outfile.close() 

    def save_Items(self,path):
         pwd=os.getcwd()
         outfile = open(path, 'w')
         for i in range(self.num_items):
            outfile.write (str(self.items[i, 0]))
            for d in range(1,self.latent_d):
                outfile.write(",")
                outfile.write (str(self.items[i, d]))
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
            
            r_hat = np.sum(users[i] * items[j])

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
                    u.append(np.sum(users[int(i)] * items[ii]))
                u.sort(reverse = True)
                r_hat = np.sum(users[int(i)] * items[j])
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

def real_ratings(file_path,bench = 0.0):
    ratings = []
    # Get ratings per user.
    pwd=os.getcwd()
    infile = open(file_path, 'r')
    for line in infile.readlines():
        f = line.rstrip('\r\n').split(",")
        if float(f[2]) > bench:
            f = (float(f[0]),float(f[1]),float(f[2]))
            ratings.append(f)
    return ratings

if __name__ == "__main__":
    if len(sys.argv) <> 7:
        print ''
        print 'Usage:  %s <file_path> \
                <latent_d_value> <beta_value> <Wm_value> \
                <output_U_filename> <output_V_filename>' % sys.argv[0]
        print ''
        sys.exit(1)
    file_path = sys.argv[1]
    latent_d_value = int(sys.argv[2])
    beta_value,Wm_value = map(float,sys.argv[3:5])
    output_U_filename, output_V_filename = sys.argv[5:7]

    ratings = real_ratings(file_path,bench = 0.0)
    pmf = ProbabilisticMatrixFactorization(ratings, latent_d=latent_d_value, beta=beta_value,Wm=Wm_value)
    iterations = 5000
    liks = []
    print "before RMSE ",pmf.rmse()
    while (pmf.update() and iterations>0):
        lik = pmf.get_current_loss()
        liks.append(lik)
        print "L=", lik
        iterations -= 1
        pass
    
    print "after RMSE ",pmf.rmse()
    #Hk,recall = pmf.topK_Hit_Ratio()
    #print Hk,recall
    pmf.save_Users(output_U_filename)
    pmf.save_Items(output_V_filename)
    #pmf.print_latent_vectors()
    #pmf.save_latent_vectors("models/")
