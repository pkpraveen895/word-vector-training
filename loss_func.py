import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    A = tf.log(tf.exp(tf.cast(tf.diag_part(tf.matmul(inputs, tf.transpose(true_w))), tf.float32)))
    #print (A)
    B = tf.log(tf.reduce_sum(tf.exp(tf.cast(tf.matmul(true_w, tf.transpose(inputs)), tf.float32)), 1, keep_dims=True))
    #print (B)
    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    
    sample_length = len(sample)
    reshaped_labels = tf.reshape(labels,[tf.shape(labels)[0]])
    
    wt_o = tf.gather(weights,reshaped_labels)
    unigram_probability_o = tf.gather(unigram_prob,reshaped_labels)
    bias_o = tf.gather(biases,reshaped_labels)
    temp_o = tf.matmul(wt_o,inputs,False,True)
    func_s_o = tf.add(bias_o,tf.matrix_diag_part(temp_o))
    log_o = tf.log(tf.multiply(unigram_probability_o,sample_length))
    log_sigma_o = tf.log_sigmoid(tf.subtract(func_s_o,log_o))

    
    wt_x = tf.gather(weights,sample)
    bias_x = tf.reshape(tf.tile(tf.gather(biases,sample),[tf.shape(reshaped_labels)[0]]),[tf.shape(reshaped_labels)[0],tf.shape(sample)[0]])
    temp_x = tf.matmul(inputs,wt_x,False,True)
    func_s_x = tf.add(bias_x,temp_x)
    unigram_probability_x = tf.reshape(tf.tile(tf.gather(unigram_prob,sample),[tf.shape(reshaped_labels)[0]]),[tf.shape(reshaped_labels)[0],tf.shape(sample)[0]])
    log_x = tf.log(tf.multiply(unigram_probability_x,len(sample)))
    sigma_x = tf.subtract(tf.ones(tf.shape(tf.sigmoid(tf.subtract(func_s_x,log_x)))),tf.sigmoid(tf.subtract(func_s_x,log_x)))+ 0.000000000001
    log_sigma_x = tf.reduce_sum(tf.log(sigma_x),1)
    
    
    totalcost = tf.negative(tf.add(log_sigma_o, log_sigma_x))
    return totalcost
