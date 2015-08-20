__author__ = 'LiuShifeng'
#!/usr/bin/python
#
# Created by Shifeng Liu (2015)
#
# An implementation of matrix factorization
#
import os
import numpy
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

        self.ratings = numpy.array(rating_tuples).astype(float)
        self.converged = False

        self.num_users = int(numpy.max(self.ratings[:, 0]) + 1)
        self.num_items = int(numpy.max(self.ratings[:, 1]) + 1)

        print (self.num_users, self.num_items, self.latent_d)
        print self.ratings

        val = []
        row = []
        col = []
        select = []
        for relation in social_tuples:
            row.append( int(relation[0]) )
            col.append( int(relation[1]) )
            val.append( float(relation[2]) )
            select.append( (int(relation[0]), int(relation[1])) )
        self.socials = sparse.csr_matrix( (val, (row, col)),shape=(self.num_users,self.num_users) )

        self.users = numpy.random.random((self.num_users, self.latent_d))
        self.items = numpy.random.random((self.num_items, self.latent_d))

        self.new_users = numpy.random.random((self.num_users, self.latent_d))
        self.new_items = numpy.random.random((self.num_items, self.latent_d))

    def loss(self, users=None, items=None, socials = None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items
        if socials is None:
            socials = self.socials

        sq_error = 0

        for rating_tuple in self.ratings:
            if len(rating_tuple) == 3:
                (i, j, rating) = rating_tuple
                weight = 1
            elif len(rating_tuple) == 4:
                (i, j, rating, weight) = rating_tuple

            r_hat_ui = numpy.sum(users[i] * items[j])
            r_hat_social = 0.0
            for v in range(self.num_users):
                r_hat_temp = numpy.sum(users[v] * items[j])
                r_hat_social += socials[i,v] * r_hat_temp
            r_hat = self.lambda_c * r_hat_ui + (1-self.lambda_c) * r_hat_social

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

        updates_u = numpy.zeros((self.num_users, self.latent_d))
        updates_v = numpy.zeros((self.num_items, self.latent_d))

        for rating_tuple in self.ratings:
            if len(rating_tuple) == 3:
                (i, j, rating) = rating_tuple
                weight = 1
            elif len(rating_tuple) == 4:
                (i, j, rating, weight) = rating_tuple
            r_hat_rating = numpy.sum(self.users[i] * self.items[j])
            r_hat_social = 0.0
            for neighbour in range(self.num_users):
                r_hat_social += self.socials[i,neighbour] * numpy.sum(self.users[neighbour] * self.items[j])
            r_hat = self.lambda_c * r_hat_rating + (1 - self.lambda_c) * r_hat_social

            for d in range(self.latent_d):
                # u contribution to u' gradient
                gradient_u = self.lambda_c * self.items[j, d] * (rating - r_hat) * weight

                # u contribution to u'neighbours'gradient
                for neighbour in range(self.num_users):
                    if self.socials[i,neighbour] == 0.0:
                        continue
                    gradient_n = (1 - self.lambda_c) * self.socials[i,neighbour] * self.items[j, d] * (rating - r_hat) * weight
                    if rating == 0.0:
                        updates_u[neighbour,d] += gradient_n * self.Wm
                    else:
                        updates_u[neighbour,d] += gradient_n

                #v contribution to v' gradient
                gradient_v = self.lambda_c * self.users[i,d]
                for neighbour in range(self.num_users):
                    gradient_v += (1 - self.lambda_c) * self.socials[i,neighbour] * self.users[neighbour,d]
                gradient_v = gradient_v * (rating - r_hat) * weight

                if rating == 0.0:
                    print "update rating 0 "
                    updates_u[i, d] += gradient_u * self.Wm
                    updates_v[j, d] += gradient_v * self.Wm
                else:
                    updates_u[i, d] += gradient_u
                    updates_v[j, d] += gradient_v

        while (not self.converged):
            initial_lik = self.loss()

            print "  setting learning rate =", self.learning_rate
            self.try_updates(updates_u, updates_v)

            final_lik = self.loss(self.new_users, self.new_items)

            if final_lik < initial_lik:
                self.apply_updates(updates_u, updates_v)
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


    def apply_updates(self, updates_u, updates_v):
        for i in range(self.num_users):
            for d in range(self.latent_d):
                self.users[i, d] = self.new_users[i, d]

        for i in range(self.num_items):
            for d in range(self.latent_d):
                self.items[i, d] = self.new_items[i, d]

    def try_updates(self, updates_u, updates_v):
        alpha = self.learning_rate

        for i in range(self.num_users):
            for d in range(self.latent_d):
                self.new_users[i,d] = self.users[i, d] + \
                                       alpha * (-self.lambda_u * self.users[i, d] + 2*updates_u[i, d])
        for i in range(self.num_items):
            for d in range(self.latent_d):
                self.new_items[i, d] = self.items[i, d] + \
                                       alpha * (-self.lambda_v * self.items[i, d] + 2*updates_v[i, d])

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

def real_ratings(bench = 0.0):
    ratings = []
    socials = []

    # Get ratings per user.
    pwd=os.getcwd()
    infile = open(os.path.join(pwd, 'u.csv'), 'r')
    for line in infile.readlines():
        f = line.rstrip('\r\n').split(",")
        if float(f[2]) > bench:
            f = (float(f[0]),float(f[1]),float(f[2]))
            ratings.append(f)
    infile.close()

    #Get social relationships.
    infile = open(os.path.join(pwd, 'u.csv'), 'r')
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

    pmf = STE(ratings, socials, latent_d=5, lambda_c = 0.1, lambda_u = 0.1, lambda_v = 0.1, Wm = 0.0)
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