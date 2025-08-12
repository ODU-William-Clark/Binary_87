The code is used to calculate mass to luminosity ratios in the K-magnitudes and B -V color indices for binary galaxies.

In particular, after galaxy binaries have been identified, the are fed through Binary_probabilites to determine the likelihood they are true binaries.

Then artificial galaxies are created using  Binary_simulate_galaxies_parallel to generate projected psi values, which are later to be compared to the real sample of galaxies in question.

BInary_87 then calculates mass to light ratios for the real galaxies for multiple differnt values of eccentricites and morphological functions.



The remaining functions are used to create realistic r_p distributions (r_distro) and Binary_psi is used for calculating psi_p values.  

IMPORTANT: This code works for the particular magnitude color band given. Things will need to change to get realistic distributions and M/L in other bands.
