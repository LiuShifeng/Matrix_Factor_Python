__author__ = 'LiuShifeng'
#!/usr/bin/python
#
# Created by Shifeng Liu (2015)
#
# An implementation of matrix factorization
#
import os
import numpy as np
import math
from scipy import sparse

class STE():

    def __init__(self, rating_tuples, social_tuples, latent_d=10, lambda_c = 0.1, lambda_u = 0.1, lambda_v = 0.1, Wm=0.0002):
        self.latent_d = latent_d
        self.learning_rate = .0001
        self.lambda_c = lambda_c
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.Wm = Wm
        print Wm

        self.ratings = np.array(rating_tuples).astype(float)
        self.socials = np.array(social_tuples).astype(float)
        self.converged = False

        self.num_users = int(np.max(self.ratings[:, 0]) + 1)
        self.num_items = int(np.max(self.ratings[:, 1]) + 1)

        print (self.num_users, self.num_items, self.latent_d)
        print self.ratings

        val = []
        row = []
        col = []
        select = []
        for rating in rating_tuples:
            row.append( int(rating[0]) )
            col.append( int(rating[1]) )
            val.append( float(rating[2]) )
            select.append( (int(rating[0]), int(rating[1])) )
        self.rating_matrix = sparse.csr_matrix( (val, (row, col)),shape=(self.num_users,self.num_items) )

        self.users = np.random.random((self.num_users, self.latent_d))
        self.items = np.random.random((self.num_items, self.latent_d))

        self.new_users = np.random.random((self.num_users, self.latent_d))
        self.new_items = np.random.random((self.num_items, self.latent_d))

        self.current_loss = self.loss()

    def loss(self, users=None, items=None, socials = None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items
        social_error = np.copy(users)

        for social_tuple in self.socials:
            if len(social_tuple) == 2:
                (i, j) = social_tuple
                rating = 1
                weight = 1
            elif len(social_tuple) == 3:
                (i, j, rating) = social_tuple
                weight = 1
            elif len(social_tuple) == 4:
                (i, j, rating, weight) = social_tuple
            social_error[i] +=  (1 - self.lambda_c) * rating * users[j] * weight / self.lambda_c

        sq_error = 0

        for rating_tuple in self.ratings:
            if len(rating_tuple) == 3:
                (i, j, rating) = rating_tuple
                weight = 1
            elif len(rating_tuple) == 4:
                (i, j, rating, weight) = rating_tuple

            r_hat = self.lambda_c * np.sum(social_error[i] * items[j])

            if rating == 0.0 :
                sq_error += self.Wm * weight * (rating - r_hat)**2
                print "Loss rating 0"
            else:
                sq_error += weight * (rating - r_hat)**2

        L2_norm = 0
        for i in range(self.num_users):
            for d in range(self.latent_d):
                L2_norm += self.lambda_u * users[i, d]**2

        for i in range(self.num_items):
            for d in range(self.latent_d):
                L2_norm += self.lambda_v * items[i, d]**2

        return sq_error + L2_norm


    def update(self):

        updates_u = np.zeros((self.num_users, self.latent_d))
        updates_v = np.zeros((self.num_items, self.latent_d))
        social_error = np.copy(self.users)
        #prepare U+sum(SU)*(1-alpha)/alpha
        for social_tuple in self.socials:
            if len(social_tuple) == 2:
                (i, j) = social_tuple
                rating = 1
                weight = 1
            elif len(social_tuple) == 3:
                (i, j, rating) = social_tuple
                weight = 1
            elif len(social_tuple) == 4:
                (i, j, rating, weight) = social_tuple
            social_error[i] +=  (1 - self.lambda_c) * rating * self.users[j] * weight / self.lambda_c

        for rating_tuple in self.ratings:
            if len(rating_tuple) == 3:
                (i, j, rating) = rating_tuple
                weight = 1
            elif len(rating_tuple) == 4:
                (i, j, rating, weight) = rating_tuple

            r_hat = self.lambda_c * np.sum(social_error[i] * self.items[j])

            for d in range(self.latent_d):
                if rating == 0.0:
                    print "update rating 0 "
                    updates_u[i, d] += self.items[j, d] * (rating - r_hat) * weight * self.lambda_c * self.Wm
                    updates_v[j, d] += social_error[i, d] * (rating - r_hat) * weight * self.lambda_c * self.Wm
                else:
                    updates_u[i, d] += self.items[j, d] * (rating - r_hat) * weight * self.lambda_c
                    updates_v[j, d] += social_error[i, d] * (rating - r_hat) * weight * self.lambda_c

        #update the social gradient
        for social_tuple in self.socials:
            if len(social_tuple) == 2:
                (i, j) = social_tuple
                rating = 1
                weight = 1
            if len(social_tuple) == 3:
                (i, j, rating) = social_tuple
                weight = 1
            elif len(social_tuple) == 4:
                (i, j, rating, weight) = social_tuple

            for item in range(self.num_items):
                gradient_j = (self.rating_matrix[i,item] - self.lambda_c * np.sum(social_error[i] * self.items[item])) * (1 - self.lambda_c) * rating * self.items[item]
                if self.rating_matrix[i,item] == 0.0:
                    updates_u[j] += gradient_j * weight * self.Wm
                else:
                    updates_u[j] += gradient_j * weight

        while (not self.converged):
            initial_lik = self.current_loss

            print "  setting learning rate =", self.learning_rate
            self.try_updates(updates_u, updates_v)

            final_lik = self.loss(self.new_users, self.new_items)

            if final_lik < initial_lik:
                self.apply_updates(updates_u, updates_v)
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


    def apply_updates(self, updates_u, updates_v):
        self.users = np.copy(self.new_users)
        self.items = np.copy(self.new_items)

    def try_updates(self, updates_u, updates_v):
        alpha = self.learning_rate

        for i in range(self.num_users):
            self.new_users[i] = self.users[i] + \
                                    alpha * (-self.lambda_u * self.users[i] + 2*updates_u[i])
        for i in range(self.num_items):
            self.new_items[i] = self.items[i] + \
                                    alpha * (-self.lambda_v * self.items[i] + 2*updates_v[i])

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

def real_ratings(bench = 0.0):
    ratings = []
    socials = []

    # Get ratings per user.
    pwd=os.getcwd()
    infile = open(os.path.join(pwd, 'ratings_data.csv'), 'r')
    for line in infile.readlines():
        f = line.rstrip('\r\n').split(",")
        if float(f[2]) > bench:
            f = (float(f[0]),float(f[1]),float(f[2]))
            ratings.append(f)
    infile.close()

    #Get social relationships.
    infile = open(os.path.join(pwd, 'trust_data.csv'), 'r')
    for line in infile.readlines():
        f = line.rstrip('\r\n').split(",")
        if len(f) == 2:
            f = (float(f[0]),float(f[1]),float(1))
        else:
            f = (float(f[0]),float(f[1]),float(f[2]))
        socials.append(f)
    infile.close()

    return ratings,socials

if __name__ == "__main__":

    ratings,socials = real_ratings(bench = 0.0)

    pmf = STE(ratings, socials, latent_d=10, lambda_c = 0.001, lambda_u = 0.001, lambda_v = 0.001, Wm = 0.0)
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
    Hk,recall = pmf.topK_Hit_Ratio()
    print Hk,recall
    '''
    pmf.save_Users('Mf\Users.data')
    pmf.save_Items('MF\Items.data')
    pmf.print_latent_vectors()

    pmf.save_latent_vectors("models/")
'''