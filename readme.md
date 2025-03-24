# max_concurrent_flow

This repo implements the algorithm described in
Section 5 from the paper:

''Faster and Simpler Algorithms for Multicommodity Flow and other Fractional Packing Problems''

Given a graph with edge capacities c
and k commodities with si ; ti being the source,
sink for commodity i. Now each commodity has a demand
d(i) associated with it and we want to find the largest 
\lambda
such that there is a multicommodity flow which routes d(i)
units of commodity i.

In particular, 
`max_concurrent_flow_nosplit` implements a modified version
in which we replace the mcf function by the sp function.

Then, we implement 
``lambda_max_concurrent_flow_nosplit``
to compute what is the lambda 


