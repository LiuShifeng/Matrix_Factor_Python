__author__ = 'LiuShifeng'
#!/usr/bin/python
#
# Created by Shifeng Liu (2015)
#
# An implementation of matrix factorization
#
import os,sys
import numpy as np
import math

class Sorec():

    def __init__(self, rating_tuples, social_tuples, latent_d=10, lambda_c = 0.1, lambda_u = 0.1, lambda_v = 0.1, lambda_z = 0.1, Wm=0.0002, Ws = 0.0002):
        self.latent_d = latent_d
        self.learning_rate = .0001
        self.lambda_c = lambda_c
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.lambda_z = lambda_z
        self.Wm = Wm
        self.Ws = Ws
        print Wm, Ws

        self.ratings = np.array(rating_tuples).astype(float)
        self.socials = np.array(social_tuples).astype(float)
        self.converged = False

        self.num_users = int(np.max(self.ratings[:, 0]) + 1)
        self.num_items = int(np.max(self.ratings[:, 1]) + 1)

        print (self.num_users, self.num_items, self.latent_d)
        print self.ratings
        print self.socials

        self.users = np.random.random((self.num_users, self.latent_d))
        self.items = np.random.random((self.num_items, self.latent_d))
        self.social = np.random.random((self.num_users, self.latent_d))

        self.new_users = np.random.random((self.num_users, self.latent_d))
        self.new_items = np.random.random((self.num_items, self.latent_d))
        self.new_social = np.random.random((self.num_users, self.latent_d))

        self.current_loss = self.loss()

    def loss(self, users=None, items=None, social = None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items
        if social is None:
            social = self.social

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

        for rating_tuple in self.socials:
            if len(rating_tuple) == 2:
                (i,j) = rating_tuple
                rating = 1
                weight = 1
            elif len(rating_tuple) == 3:
                (i, j, rating) = rating_tuple
                weight = 1
            elif len(rating_tuple) == 4:
                (i, j, rating, weight) = rating_tuple

            r_hat = np.sum(users[i] * social[j])
            if rating == 0.0 :
                sq_error += self.Ws * weight * self.lambda_c * (rating - r_hat)**2
            else:
                sq_error += weight * self.lambda_c * (rating - r_hat)**2

        L2_norm = 0
        L2_norm += self.lambda_u * np.sum(users*users)
        L2_norm += self.lambda_v * np.sum(items*items)
        L2_norm += self.lambda_z * np.sum(socials*socials)

        return sq_error + L2_norm


    def update(self):

        updates_u = np.zeros((self.num_users, self.latent_d))
        updates_v = np.zeros((self.num_items, self.latent_d))
        updates_z = np.zeros((self.num_users, self.latent_d))

        for rating_tuple in self.ratings:
            if len(rating_tuple) == 3:
                (i, j, rating) = rating_tuple
                weight = 1
            elif len(rating_tuple) == 4:
                (i, j, rating, weight) = rating_tuple

            r_hat = np.sum(self.users[i] * self.items[j])

            if rating == 0.0:
                updates_u[i] += self.items[j] * (rating - r_hat) * weight * self.Wm
                updates_v[j] += self.users[i] * (rating - r_hat) * weight * self.Wm
            else:
                updates_u[i] += self.items[j] * (rating - r_hat) * weight
                updates_v[j] += self.users[i] * (rating - r_hat) * weight

        for rating_tuple in self.socials:
            if len(rating_tuple) == 2:
                (i,j) = rating_tuple
                rating = 1
                weight = 1
            elif len(rating_tuple) == 3:
                (i, j, rating) = rating_tuple
                weight = 1
            elif len(rating_tuple) == 4:
                (i, j, rating, weight) = rating_tuple

            r_hat = np.sum(self.users[i] * self.social[j])

            if rating == 0.0:
                updates_u[i] += self.social[j] * (rating - r_hat) * weight * self.lambda_c * self.Ws
                updates_z[j] += self.users[i] * (rating - r_hat) * weight * self.lambda_c * self.Ws
            else:
                updates_u[i] += self.social[j] * (rating - r_hat) * weight * self.lambda_c
                updates_z[j] += self.users[i] * (rating - r_hat) * weight * self.lambda_c


        while (not self.converged):
            initial_lik = self.current_loss

            print "  setting learning rate =", self.learning_rate
            self.try_updates(updates_u, updates_v,updates_z)

            final_lik = self.loss(self.new_users, self.new_items, self.new_social)

            if final_lik < initial_lik:
                self.apply_updates(updates_u, updates_v,updates_z)
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


    def apply_updates(self, updates_u, updates_v, updates_z):
        self.users = np.copy(self.new_users)
        self.items = np.copy(self.new_items)
        self.social =  np.copy(self.new_social)

    def try_updates(self, updates_u, updates_v, update_z):
        alpha = self.learning_rate

        for i in range(self.num_users):
            self.new_users[i] = self.users[i,] + \
                                    alpha * (-self.lambda_u * self.users[i] + 2*updates_u[i])
        for i in range(self.num_items):
            self.new_items[i] = self.items[i] + \
                                    alpha * (-self.lambda_v * self.items[i] + 2*updates_v[i])
        for i in range(self.num_users):
            self.new_social[i] = self.social[i] + \
                                    alpha * (-self.lambda_u * self.social[i] + 2*update_z[i])

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

        print "Social"
        for i in range(self.num_users):
            print i,
            for d in range(self.latent_d):
                print self.social[i, d],
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

    def print_Social(self):
        print "Social"
        for i in range(self.num_users):
            print i,
            for d in range(self.latent_d):
                print self.social[i, d],
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

    def save_Social(self,path):
         pwd=os.getcwd()
         outfile = open(path, 'w')
         for i in range(self.num_users):
            outfile.write (str(self.social[i, 0]))
            for d in range(1,self.latent_d):
                outfile.write(",")
                outfile.write (self.social[i, d])
            outfile.write ('\n')
         outfile.close()

    def save_latent_vectors(self, prefix):
        self.users.dump(prefix + "%sd_users.pickle" % self.latent_d)
        self.items.dump(prefix + "%sd_items.pickle" % self.latent_d)
        self.social.dump(prefix + "%sd_social.pickle" % self.latent_d)

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

def real_ratings(rating_path,social_path,bench = 0.0):
    ratings = []
    socials = []

    # Get ratings per user.
    pwd=os.getcwd()
    infile = open(rating_path, 'r')
    for line in infile.readlines():
        f = line.rstrip('\r\n').split(",")
        if float(f[2]) > bench:
            f = (float(f[0]),float(f[1]),float(f[2]))
            ratings.append(f)
    infile.close()

    #Get social relationships.
    infile = open(social_path, 'r')
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
    if len(sys.argv) <> 12:
        print ''
        print 'Usage:  %s <rating_file_path> <social_file_path>\
                <latent_d_value> <lambda_c_value> <lambda_u_value> <lambda_v_value> <lambda_z_value> <Wm_value> <Ws_value>\
                <output_U_filename> <output_V_filename>' % sys.argv[0]
        print ''
        sys.exit(1)
    rating_file_path,social_file_path = sys.argv[1:3]
    latent_d_value = int(sys.argv[3])
    lambda_c_value,lambda_u_value,lambda_v_value,lambda_z_value,Wm_value,Ws_value = map(float,sys.argv[4:10])
    output_U_filename, output_V_filename = sys.argv[10:12]

    ratings,socials = real_ratings(rating_file_path,social_file_path,bench = 0.0)

    pmf = Sorec(ratings, socials, latent_d=latent_d_value, lambda_c = lambda_c_value, lambda_u = lambda_u_value, lambda_v = lambda_v_value, lambda_z = lambda_z_value, Wm = Wm_value, Ws = Ws_value)
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