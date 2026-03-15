import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim=128):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.W_in = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))
        self.W_out = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))

    def forward_backward(self, center_words, context_words, negative_words, learning_rate=0.01):
        """
        Performs a batched forward pass to calculate loss,
        and a backward pass to update weights via SGD
        """
        v_c = self.W_in[center_words]            
        u_p = self.W_out[context_words]          
        u_n = self.W_out[negative_words]         

        # FORWARD PASS
        score_pos = np.sum(v_c * u_p, axis=1)    # Shape: (batch_size,)
        
        # Dot product for negative pairs. We reshape v_c to broadcast across num_neg
        score_neg = np.sum(v_c[:, np.newaxis, :] * u_n, axis=2) # Shape: (batch_size, num_neg)

        # Apply sigmoid
        prob_pos = sigmoid(score_pos)            # Shape: (batch_size,)
        prob_neg = sigmoid(score_neg)            # Shape: (batch_size, num_neg)

        eps = 1e-7
        loss_pos = -np.log(prob_pos + eps)
        loss_neg = -np.sum(np.log(1 - prob_neg + eps), axis=1)
        batch_loss = np.mean(loss_pos + loss_neg)

        # BACKWARD PASS
        grad_score_pos = prob_pos - 1.0          # Target is 1 (Positive)
        grad_score_neg = prob_neg                # Target is 0 (Negative)

        grad_u_p = grad_score_pos[:, np.newaxis] * v_c   
        grad_u_n = grad_score_neg[:, :, np.newaxis] * v_c[:, np.newaxis, :] 

        grad_v_c_pos = grad_score_pos[:, np.newaxis] * u_p 
        grad_v_c_neg = np.sum(grad_score_neg[:, :, np.newaxis] * u_n, axis=1) 
        grad_v_c = grad_v_c_pos + grad_v_c_neg             


        # SGD

        np.add.at(self.W_in, center_words, -learning_rate * grad_v_c)
        np.add.at(self.W_out, context_words, -learning_rate * grad_u_p)
        
        np.add.at(self.W_out, negative_words.flatten(), 
                  -learning_rate * grad_u_n.reshape(-1, self.embedding_dim))

        return batch_loss