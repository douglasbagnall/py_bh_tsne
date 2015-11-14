This is a Python wrapper for [Barnes-Hut
t-SNE](http://homepage.tudelft.nl/19j49/t-SNE.html) forked from
[Christian Osendorfer's version](https://github.com/osdf/py_bh_tsne).
It has been tuned with specific uses in mind, so it *might* better
than the original for you, and if it does, it is likely to be slightly
faster.


License
-------

This (and other forks) are based on Laurens van der Maaten's original
[code](http://homepage.tudelft.nl/19j49/t-SNE.html) which is now licensed
under a [4 clause BSD
license](https://github.com/lvdmaaten/bhtsne/blob/master/LICENSE.txt).
Previously it was “for non-commercial purposes only”, and this
language is still used in some places.

My modifications can be used under the same terms as the upstream
t-SNE, whatever they are at any particular time.

This product includes software developed by the Delft University of
Technology.


Changes from the osdf version
-----------------------------

* Requires with `cblas`, not `openblas`.
* Optimised compilation arguments, including `-ffast-math`.
* All unused function in original code are removed.
* Exact t-SNE option is gone.
* Options for normalising vectors to unit hyper-sphere.
* Fiddling with pre-t-SNE PCA options.


Requirements
------------

* [numpy](numpy.scipy.org)
* [cython](cython.org)


Testing
-------

For testing the algorithm, add ```fasttsne/``` to your
```PYTHONPATH``` and run ```python test.py``` after a successful
build. Note that the file ```mnist.pkl.gz``` has to be in the main
directory. You can download it from
[here](http://deeplearning.net/data/mnist/mnist.pkl.gz).


More Information
----------------

See *Barnes-Hut-SNE*, L.J.P. van der Maaten. It is available on
[arxiv](http://arxiv.org/abs/1301.3342). Also check out Christian
Osendorfer's version, which is careful not to mess too much with the
original code.
