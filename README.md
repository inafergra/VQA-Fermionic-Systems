Numerical simulation of a cooling algorithm for fermionic systems. 

Issues:
- For the SYK: energy of the algorithm goes below the exact ground state energy. Either the exact energy is calculated wrongly or some piece of the cooling algorithm is not correct. 

Things to check: 

- constants in the Hamiltonian
- normalization of the initial state
- matrix elements (calculated with Mathematica)
- exact diagonalization
- Check the symmetry of the SYK coupling matrices

Code has a lot of function calls that slow down the execution.