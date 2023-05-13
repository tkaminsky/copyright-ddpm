import torch
from torch.distributions.multivariate_normal import MultivariateNormal


# Implement the CP-k algorithm for our model
class cp_k_model():
    # Pass a network and a list of models that cover the copyrighted data
    def __init__(self, net, cover, shape, k=500, T=1000):
        # The model that we are trying to protect
        self.net = net
        # A list of models that cover the data
        self.cover = cover
        # The shape of the image data (b x c x h x w)
        self.shape = shape
        # Our k threshold
        self.k = 500
        self.T = T

    def eval(self):
        self.net.eval()
        for model in self.cover:
            model.eval()
    
    def train(self):
        self.net.train()
        for model in self.cover:
            model.train()
        

    # Calculates the log probability of a transition for each x
    def get_transition_logprob(self, mean, log_var, x_t):
        # The conditional multivariate normal distribution
        dist = MultivariateNormal(mean, torch.exp(log_var))
        log_p = dist.log_prob(x_t)
        return log_p

    def get_vectorized_transition_logprob(self, mean, log_var, X_t):
        # The conditional multivariate normal distribution
        logprobs = torch.zeros(X_t.shape[0], 1, device=X_t.device)
        for i in range(X_t.shape[0]):
            var_matrix = torch.eye(3072).cuda() * torch.exp(log_var[i][0][0][0])
            dist = MultivariateNormal(torch.flatten(mean[i]), var_matrix)
            logprobs[i] = dist.log_prob(torch.flatten(X_t[i]))
        return logprobs

    # Generate a guess for the initial x_0 given x_T, accumulating
    #   the log probabilities of each guess along the way
    def run_w_logprobs(self, x_T):
        x_t = x_T

        # The log probability of the net for each x
        net_logp = torch.zeros(self.shape[0], 1).cuda()
        # The log probability of each cover model for each x
        cover_logps = torch.zeros(len(self.cover), self.shape[0], 1).cuda()

        for time_step in reversed(range(self.T)):
            if time_step % 5 == 0:
                print(f"On step {time_step}")
            # Get the time step
            t = x_t.new_ones([self.shape[0], ], dtype=torch.long) * time_step

            # Get net means and log vars
            net_mean, net_log_var = self.net.p_mean_variance(x_t=x_t.cuda(), t=t.cuda())
            

            # Do the same for the cover models
            cover_means = []
            cover_log_vars = []
            for cover_model in self.cover:
                cover_mean, cover_log_var = cover_model.p_mean_variance(x_t=x_t.cuda(), t=t.cuda())
                cover_means.append(cover_mean)
                cover_log_vars.append(cover_log_var)
            
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t).cuda()
            else:
                noise = 0
            x_t = net_mean + torch.exp(0.5 * net_log_var) * noise

            # Calculate P(x_t_new | x_t) and add to the running log probability
            net_logp += self.get_vectorized_transition_logprob(net_mean, net_log_var, x_t)
            for i in range(len(self.cover)):
                cover_logps[i] += self.get_vectorized_transition_logprob(cover_means[i], cover_log_vars[i], x_t)


        x_0 = x_t
        return torch.clip(x_0, -1, 1), net_logp, cover_logps



    # Given the probabilities of each transition P(x_t | x_{t-1}), calculate the log joint probability of the chain
    def joint_pdf(self, transitions):
        log_probs = torch.log(transitions)
        log_p = log_probs.sum(dim=0)
        print(log_p)
        return log_p
    


    def sample(self):
        print("Sampling now:")
        good = 0
        good_ims = torch.zeros(self.shape)
        tries = 0
        batch_orig = self.shape[0]
        while True:
            # Generate random noise
            x_0, net_logp, cover_logps = self.run_w_logprobs(x_T = torch.randn(self.shape))

            # If the difference between the net log probability and the cover log probability is less than k, return the sample

            # A B x 1 tensor of the difference between the net log probability and the cover log probability
            diff_a = net_logp - cover_logps[0]
            diff_b = net_logp - cover_logps[1]

            for i in range(diff_a.shape[0]):
                if diff_a[i] < self.k and diff_b[i] < self.k:
                    good += 1
                    good_ims[i] = x_0[i]
                    self.shape[0] -= 1
            
            print(max(diff_a))
            print(max(diff_b))

            if good == batch_orig:
                # All the samples are valid!
                print(f"Succeeded in generating sample in {tries} tries")
                self.shape[0] = batch_orig
                return good_ims
            if tries > 100:
                print("Failed to generate sample in 100 tries")
                self.shape[0] = batch_orig
                return None
            tries += 1
            print(f"On try {tries}; current shape: {self.shape}")
