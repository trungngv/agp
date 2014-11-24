### Automated Variational Inference for Gaussian Process models.

Code for the paper "Automated Variational Inference for Gaussian Process Models". If you use any of the code, please cite:

Nguyen, T.V. and Bonilla, E.V., Automated Variational Inference for Gaussian Process Models, In *NIPS* 2014.

Please see **src/demoFull.m** and **src/demoMixture.m** for an example of how to use the code to perform inference for the regression model with the full Gaussian distribution and the mixture of Gausians distribution, respectively. The code also includes implementation of other models: binary classification, multi-class classification, warped Gaussian processes, and log Gaussian Cox process. Please see the **src/likelihood** directory for details. The predictions for these models are also implemented in **src/prediction**.

To try a new model, simply implement a new likelihood function following the signature:

      logllh = newFunction(y,f,hyp)
where 

      y : N x P vector of observations (each column corresponds to one output; N is the number of inputs)
      f : N x Q matrix of latent function values (each column corresponds to one latent function; Q is the number of latent functions)       

