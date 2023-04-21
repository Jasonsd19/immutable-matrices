# immutable-matrices
A basic linear algebra matrix library, written in C++.

The goal was to create a 'reasonably' performant library for matrix calculations. To this end I've utilized concepts such as matrix blocking, cache locality and temporality maximization, and multi-threading (using pthread). The main fuctions of importance for these techniques are the `transpose()` and `matrixMultiply()` functions as they best show-case the aforementioned concepts, and were the main focus of performance optimizations.

Additionally, I've also utilized PLU factorization/decomposition as the basis for the various other foundational matrix functions such as `LUPDecompose()`, `solve()`, `determinant()`, and `inverse()`. The benefit to using PLU factorization is two-fold. First, it is a numerically stable algorithm which leads to less errors in our calculations. Second, it allows users to retain the calculated Lower and Upper Triangular matrices and re-use them in future calculations. The benefit to this is that unlike other algorithms (ie. naive Gaussian Elimination), future calculations require less overhead because a lot of the work was in solving for the Lower and Upper Triangular matrices which we now have readily on hand.

As of the time of writing this README there are some limitations to this library. First, the `LUPDecompose()` function only accepts square matrices. Second, the `solve()` function only returns unique solutions, and will throw an error otherwise. If I return to this project I would like to add additional functionality that would allow for the PLU factorization of certain non-square matrices, and allow for better differentiation between inconsistent/consistent systems of linear equations.

I've also written my own basic testing framework for this project, and I definitely learned a lot of interesting C++ concepts along the way (comparing doubles is tough!).

To run the tests on Linux you will need to have installed make, cmake, and a C++ compiler. Afterwards:

* Clone the project
* run `cd ./immutable-matrices`
* run `make build`
* run `make test`

Then you should (hopefully) see the passing tests print out to the terminal.

I've provide a decent ammount of comments in the code so please look there first if you have any questions. If your question is still not answered then please feel free to reach out to me.
