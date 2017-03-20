# z4_planar_langford
CPU (multicore) code to count all PLANAR Langford sequences.

Copyright 2017 Boris Dimitrov, Portola Valley, CA 94028.

Explanation and code:

    planar_mt.cpp
    
Execution log for the computation of PL(2, 27):

    PL_2_27_computation_on_22_core_xeon_E5-2699v4.log
    
CPU info for the 22-core Xeon E5-2699v4 (14nm, early 2016) on which the
PL(2, 27) computation took place:

    cpu_info_22_core_xeon_E5-2699v4.txt
    
See also the GPU version of this code:

    https://github.com/boris-dimitrov/z4_planar_langford_multigpu
    
As of 2017-03-19 this algorithm is outperformed by the technique in

    https://github.com/boris-dimitrov/z5_langford/blob/master/langford.cpp
    
which appears to produce results in 30% less time.  Both algorithms have
exponential complexity O(3.38^n).

Questions? Contact http://www.facebook.com/boris
