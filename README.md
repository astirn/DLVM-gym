Here are two (what I beleive to be) equivalent implementations of a VAE with a VampPrior.
Only twice in my life, have I seen a VAE get stuck at a poor optima at the onset of training (e.g. posterior collapse). 
Both times were using torch. I am now convinced something is wrong.
The tensorflow version consistently achieves an ELBO > 300 and the torch version consistently attains < -600. What gives? Anyone?
